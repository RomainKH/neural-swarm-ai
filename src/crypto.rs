use aes_gcm::{
    aead::{Aead, KeyInit, Payload},
    Aes256Gcm, Nonce,
};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use rand::{rngs::OsRng, RngCore};
use sha2::{Digest, Sha256};
use x25519_dalek::{PublicKey, StaticSecret};

// Create alias for HMAC-SHA256
type HmacSha256 = Hmac<Sha256>;

/// Generates a random 32-byte nonce for the AuthChallenge.
pub fn generate_nonce() -> [u8; 32] {
    let mut nonce = [0u8; 32];
    OsRng.fill_bytes(&mut nonce);
    nonce
}

/// Signs a nonce using HMAC-SHA256 and the shared secret.
pub fn sign_hmac(nonce: &[u8; 32], secret: &str) -> Vec<u8> {
    let mut mac = <HmacSha256 as KeyInit>::new_from_slice(secret.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(nonce);
    mac.finalize().into_bytes().to_vec()
}

/// Verifies a received hash against the expected HMAC-SHA256.
pub fn verify_hmac(nonce: &[u8; 32], secret: &str, received_hash: &[u8]) -> bool {
    let mut mac = <HmacSha256 as KeyInit>::new_from_slice(secret.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(nonce);
    mac.verify_slice(received_hash).is_ok()
}

// ── Perfect Forward Secrecy (ECDH) ───────────────────────────

/// Generates an ephemeral X25519 key pair.
pub fn generate_ecdh_keys() -> (StaticSecret, [u8; 32]) {
    let secret = StaticSecret::random_from_rng(OsRng);
    let public = PublicKey::from(&secret);
    (secret, *public.as_bytes())
}

/// Derives a shared session key using ECDH and HKDF.
pub fn derive_session_key(
    my_secret: &StaticSecret,
    their_public: &[u8; 32],
    salt: &[u8],
) -> [u8; 32] {
    let their_public = PublicKey::from(*their_public);
    let shared_secret = my_secret.diffie_hellman(&their_public);

    let hk = Hkdf::<Sha256>::new(Some(salt), shared_secret.as_bytes());
    let mut okm = [0u8; 32];
    hk.expand(b"neural-swarm-ai-v0.2-session", &mut okm)
        .expect("32 bytes is valid for SHA256");
    okm
}

// ── Compression & Encryption ──────────────────────────────────

/// Compresses data using zstd.
pub fn compress(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    zstd::bulk::compress(data, 3)
}

/// Decompresses data using zstd.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    // Limit to 256MB to prevent zip bombs
    zstd::bulk::decompress(data, 256 * 1024 * 1024)
}

/// Encrypts a payload using AES-256-GCM.
/// Bind the encryption to Additional Authenticated Data (AAD) for replay protection.
pub fn encrypt_with_aad(data: &[u8], key: &[u8; 32], aad: &[u8]) -> Result<Vec<u8>, String> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| e.to_string())?;

    let mut nonce_bytes = [0u8; 12];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let payload = Payload { msg: data, aad };

    let ciphertext = cipher
        .encrypt(nonce, payload)
        .map_err(|e| format!("Encryption failed: {}", e))?;

    let mut result = Vec::with_capacity(12 + ciphertext.len());
    result.extend_from_slice(&nonce_bytes);
    result.extend_from_slice(&ciphertext);
    Ok(result)
}

/// Decrypts a payload using AES-256-GCM with AAD verification.
pub fn decrypt_with_aad(data: &[u8], key: &[u8; 32], aad: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 12 {
        return Err("Data too short for decryption".into());
    }

    let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| e.to_string())?;

    let (nonce_bytes, ciphertext) = data.split_at(12);
    let nonce = Nonce::from_slice(nonce_bytes);

    let payload = Payload {
        msg: ciphertext,
        aad,
    };

    cipher
        .decrypt(nonce, payload)
        .map_err(|e| format!("Decryption failed: {}", e))
}

/// Legacy wrapper for backward compatibility or simple encryption.
pub fn encrypt(data: &[u8], secret: &str) -> Result<Vec<u8>, String> {
    let key_hash = Sha256::digest(secret.as_bytes());
    let mut key = [0u8; 32];
    key.copy_from_slice(&key_hash);
    encrypt_with_aad(data, &key, &[])
}

/// Legacy wrapper for backward compatibility or simple decryption.
pub fn decrypt(data: &[u8], secret: &str) -> Result<Vec<u8>, String> {
    let key_hash = Sha256::digest(secret.as_bytes());
    let mut key = [0u8; 32];
    key.copy_from_slice(&key_hash);
    decrypt_with_aad(data, &key, &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_hmac() {
        let secret = "super_secret_token";
        let nonce = generate_nonce();

        let hash = sign_hmac(&nonce, secret);
        assert!(verify_hmac(&nonce, secret, &hash));
    }

    #[test]
    fn test_ecdh_handshake() {
        let salt = b"handshake-salt";
        let (s1, p1) = generate_ecdh_keys();
        let (s2, p2) = generate_ecdh_keys();

        let k1 = derive_session_key(&s1, &p2, salt);
        let k2 = derive_session_key(&s2, &p1, salt);

        assert_eq!(k1, k2);
    }

    #[test]
    fn test_encryption_with_aad() {
        let key = [0u8; 32];
        let data = b"Sensitive Data";
        let aad = b"task-123";

        let encrypted = encrypt_with_aad(data, &key, aad).unwrap();

        // Decrypt with correct AAD
        let decrypted = decrypt_with_aad(&encrypted, &key, aad).unwrap();
        assert_eq!(data, decrypted.as_slice());

        // Decrypt with WRONG AAD (should fail)
        let failed = decrypt_with_aad(&encrypted, &key, b"wrong-task");
        assert!(failed.is_err());
    }

    #[test]
    fn test_compression_roundtrip() {
        let data = b"Hello world! NeuralSwarmAI is awesome. ".repeat(10);
        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed.as_slice());
        assert!(compressed.len() < data.len());
    }
}
