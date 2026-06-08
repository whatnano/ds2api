'use strict';

const crypto = require('crypto');

function signingEnabled() {
  return String(process.env.DS2API_CLIENT_SIGNING_KEY || '').trim() !== '';
}

function generateClientNonce() {
  return crypto.randomBytes(16).toString('hex');
}

function buildSignaturePayload(method, pathname, timestamp, nonce) {
  return String(method || 'POST').toUpperCase().trim() + '\n' +
    String(pathname || '/') + '\n' +
    String(timestamp) + '\n' +
    String(nonce);
}

function computeClientSignature(key, payload) {
  return crypto.createHmac('sha256', key).update(payload).digest('hex');
}

function injectClientSignatureHeaders(headers, method, url) {
  if (!signingEnabled()) {
    return;
  }
  let pathname = '/';
  try {
    pathname = new URL(url).pathname || '/';
  } catch (_err) {
    // Fall through to default.
  }
  const key = String(process.env.DS2API_CLIENT_SIGNING_KEY).trim();
  const nonce = generateClientNonce();
  const ts = Date.now();
  const payload = buildSignaturePayload(method, pathname, ts, nonce);
  const sig = computeClientSignature(key, payload);

  headers['X-Client-Nonce'] = nonce;
  headers['X-Client-Timestamp'] = String(ts);
  headers['X-Client-Signature'] = sig;
}

module.exports = {
  signingEnabled,
  injectClientSignatureHeaders,
};
