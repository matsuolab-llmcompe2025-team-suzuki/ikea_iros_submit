#!/usr/bin/env bash
# Generate a private CA and AVP-compatible TLS leaf certificate.
#
# The generated rootCA.pem (public only) must be installed and explicitly
# trusted on Apple Vision Pro.  Do not copy rootCA.key off this Desktop.
set -euo pipefail

CERT_DIR="${XR_TELEOP_CERT_DIR:-$HOME/.config/xr_teleoperate_avp}"
DESKTOP_IP="${XR_DESKTOP_IP:-192.168.29.175}"
ORIN_IP="${XR_ORIN_IP:-192.168.29.159}"

if [[ -e "$CERT_DIR/rootCA.key" || -e "$CERT_DIR/cert.pem" || -e "$CERT_DIR/key.pem" ]]; then
  echo "ERROR: refusing to overwrite an existing AVP certificate directory: $CERT_DIR" >&2
  exit 1
fi

umask 077
mkdir -p "$CERT_DIR"
openssl genrsa -out "$CERT_DIR/rootCA.key" 2048
openssl req -x509 -new -nodes -key "$CERT_DIR/rootCA.key" -sha256 -days 365 \
  -out "$CERT_DIR/rootCA.pem" -subj "/CN=iros-g1-avp-root-ca"
openssl genrsa -out "$CERT_DIR/key.pem" 2048
openssl req -new -key "$CERT_DIR/key.pem" -out "$CERT_DIR/server.csr" \
  -subj "/CN=iros-g1-avp" \
  -addext "subjectAltName=DNS:localhost,IP:${DESKTOP_IP},IP:${ORIN_IP}"
openssl x509 -req -in "$CERT_DIR/server.csr" -CA "$CERT_DIR/rootCA.pem" \
  -CAkey "$CERT_DIR/rootCA.key" -CAcreateserial -out "$CERT_DIR/cert.pem" \
  -days 365 -sha256 -copy_extensions copy
chmod 600 "$CERT_DIR/rootCA.key" "$CERT_DIR/key.pem"
chmod 644 "$CERT_DIR/rootCA.pem" "$CERT_DIR/cert.pem"

echo "Created AVP certificate set in: $CERT_DIR"
echo "AirDrop this public CA file to Apple Vision Pro, install it, then enable trust:"
echo "  $CERT_DIR/rootCA.pem"
