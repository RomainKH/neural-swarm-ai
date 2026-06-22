//! Stream a GGUF model over HTTP range requests.
//!
//! candle reads a GGUF tensor-by-tensor through a `Read + Seek`. By backing that
//! reader with HTTP range requests, a node fetches ONLY the bytes of the layer
//! slice it was assigned — never the whole file. Combined with partial layer
//! loading, this lets someone contribute compute to a huge model without ever
//! installing it locally.
//!
//! Uses `ureq` (synchronous, no async runtime) so it is safe to call from inside
//! a Tokio worker without the "runtime within a runtime" panic that
//! `reqwest::blocking` would cause.

use std::io::{self, Read, Seek, SeekFrom};

/// Minimum bytes to fetch per range request — collapses the GGUF header's many
/// tiny sequential reads into a couple of requests.
const MIN_CHUNK: u64 = 4 * 1024 * 1024;

pub struct HttpRangeReader {
    url: String,
    pos: u64,
    len: u64,
    buf: Vec<u8>,
    buf_start: u64,
}

fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e.to_string())
}

impl HttpRangeReader {
    /// Opens a remote GGUF, discovering its total size up front (HEAD request).
    pub fn new(url: &str) -> io::Result<Self> {
        let resp = ureq::head(url).call().map_err(io_err)?;
        let len: u64 = resp
            .header("Content-Length")
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| io_err("remote model: missing/!invalid Content-Length"))?;
        Ok(Self {
            url: url.to_string(),
            pos: 0,
            len,
            buf: Vec::new(),
            buf_start: 0,
        })
    }

    /// Fetches a range starting at `start` of at least `want` bytes into the buffer.
    fn fetch(&mut self, start: u64, want: u64) -> io::Result<()> {
        let end = (start + want.max(MIN_CHUNK)).min(self.len); // exclusive
        if end <= start {
            self.buf.clear();
            self.buf_start = start;
            return Ok(());
        }
        let resp = ureq::get(&self.url)
            .set("Range", &format!("bytes={}-{}", start, end - 1))
            .call()
            .map_err(io_err)?;
        let mut data = Vec::with_capacity((end - start) as usize);
        resp.into_reader().read_to_end(&mut data)?;
        self.buf = data;
        self.buf_start = start;
        Ok(())
    }
}

impl Read for HttpRangeReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.len || out.is_empty() {
            return Ok(0);
        }
        let in_buf =
            self.pos >= self.buf_start && self.pos < self.buf_start + self.buf.len() as u64;
        if !in_buf {
            self.fetch(self.pos, out.len() as u64)?;
        }
        let offset = (self.pos - self.buf_start) as usize;
        if offset >= self.buf.len() {
            return Ok(0);
        }
        let avail = &self.buf[offset..];
        let n = avail.len().min(out.len());
        out[..n].copy_from_slice(&avail[..n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for HttpRangeReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new = match pos {
            SeekFrom::Start(s) => s as i64,
            SeekFrom::End(e) => self.len as i64 + e,
            SeekFrom::Current(c) => self.pos as i64 + c,
        };
        if new < 0 {
            return Err(io_err("seek before start"));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}
