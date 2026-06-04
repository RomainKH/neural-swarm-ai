use hmac::{Hmac, Mac};
use rand::{rngs::OsRng, RngCore};
use sha2::Sha256;

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
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC can take key of any size");
    mac.update(nonce);
    mac.finalize().into_bytes().to_vec()
}

/// Verifies a received hash against the expected HMAC-SHA256.
pub fn verify_hmac(nonce: &[u8; 32], secret: &str, received_hash: &[u8]) -> bool {
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC can take key of any size");
    mac.update(nonce);
    mac.verify_slice(received_hash).is_ok()
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
    fn test_invalid_hmac() {
        let secret = "super_secret_token";
        let wrong_secret = "bad_token";
        let nonce = generate_nonce();

        let hash = sign_hmac(&nonce, wrong_secret);
        assert!(!verify_hmac(&nonce, secret, &hash));
    }
}
