import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    K, M,
    BANDWIDTH_MIN_HZ, BANDWIDTH_MAX_HZ,
    TX_POWER_MIN_W, TX_POWER_MAX_W,
    NOISE_DENSITY_DBM,
)


class ChannelModel:
    """
    Wireless channel model for K IoT devices communicating with M edge nodes.

    Models:
      - Rayleigh fading (small-scale)
      - Distance-dependent path loss (large-scale)
      - Thermal noise from spectral density
      - SNR and Shannon capacity per (device, edge) link
    """

    def __init__(self, rng: np.random.Generator = None):
        """
        Parameters
        ----------
        rng : np.random.Generator, optional
            Seeded generator for reproducibility. Defaults to a fresh Generator.
        """
        self.rng = rng if rng is not None else np.random.default_rng()

        # Sample per-device transmit power once (fixed per episode)
        # Shape: (K,) — each device has its own power budget
        self.tx_power = self.rng.uniform(TX_POWER_MIN_W, TX_POWER_MAX_W, size=K)

        # Sample per-link bandwidth once (fixed per episode)
        # Shape: (K, M) — each (device, edge) link may use a different sub-band
        self.bandwidth = self.rng.uniform(BANDWIDTH_MIN_HZ, BANDWIDTH_MAX_HZ, size=(K, M))

        # Sample path-loss exponent per link (urban=2, NLOS up to 4)
        # Shape: (K, M)
        self.alpha = self.rng.uniform(2.0, 4.0, size=(K, M))

        # Sample distances between each device and each edge node (metres)
        # Shape: (K, M)  — range [10 m, 500 m]
        self.distance = self.rng.uniform(10.0, 500.0, size=(K, M))

        # Noise power N0 (W) — computed once from spectral density constant
        # NOISE_DENSITY_DBM is in dBm/Hz → convert to W/Hz → scale by bandwidth
        # N0 [W] = 10^((dBm/Hz) / 10) * 1e-3 * bandwidth [Hz]
        # We store the per-Hz value and multiply at SNR time so it stays general.
        noise_dbm_per_hz = NOISE_DENSITY_DBM                      # dBm/Hz
        noise_w_per_hz   = (10 ** (noise_dbm_per_hz / 10)) * 1e-3 # W/Hz
        # Shape: (K, M) — noise floor scales with the allocated bandwidth
        self.N0 = noise_w_per_hz * self.bandwidth                  # W

        # Placeholders — populated by step()
        self.H    = None   # Rayleigh fading gains,  shape (K, M)
        self.PL   = None   # Path loss factors,      shape (K, M)
        self.SNR  = None   # Signal-to-noise ratio,  shape (K, M)
        self.rate = None   # Shannon capacity (bps),  shape (K, M)

        # Run one step so arrays are initialised before first use
        self.step()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rayleigh_gain(self) -> np.ndarray:
        """
        Rayleigh fading channel gain h[k,m].

        For Rayleigh fading the envelope |h| follows a Rayleigh distribution,
        which is obtained as the magnitude of a complex circular-symmetric
        Gaussian:
            h = (X + jY) / sqrt(2),  X,Y ~ N(0,1)
            |h|^2 ~ Exp(1)  (unit mean power)

        Returns shape (K, M) of real-valued |h| (envelope).
        """
        real = self.rng.standard_normal(size=(K, M))
        imag = self.rng.standard_normal(size=(K, M))
        # Envelope magnitude
        h = np.sqrt(real**2 + imag**2) / np.sqrt(2)
        return h                                                    # shape (K, M)

    def _path_loss(self) -> np.ndarray:
        """
        Large-scale path loss PL[k,m].

            PL[k,m] = distance[k,m]^(-alpha[k,m])

        A higher alpha (closer to 4) represents more obstructed environments.

        Returns shape (K, M), dimensionless attenuation factor (≤ 1).
        """
        return self.distance ** (-self.alpha)                       # shape (K, M)

    def _snr(self) -> np.ndarray:
        """
        Received SNR at edge node m from device k.

            SNR[k,m] = (P_k * |h[k,m]|^2 * PL[k,m]) / N0[k,m]

        where:
            P_k      — transmit power of device k  (W)
            h[k,m]   — Rayleigh fading envelope
            PL[k,m]  — path loss attenuation
            N0[k,m]  — noise power in allocated bandwidth  (W)

        Returns shape (K, M), linear (not dB).
        """
        # tx_power is (K,) → broadcast to (K, M)
        P = self.tx_power[:, np.newaxis]                           # (K, 1)
        return (P * self.H**2 * self.PL) / self.N0                 # (K, M)

    def _shannon_rate(self) -> np.ndarray:
        """
        Shannon channel capacity (achievable data rate) in bits/second.

            R[k,m] = B[k,m] * log2(1 + SNR[k,m])

        where B[k,m] is the allocated bandwidth in Hz.

        A minimum floor of 1 Mbps is enforced so that deep Rayleigh fades
        (H → 0, SNR → 0) do not produce near-zero rates, which would cause
        transmission delays of thousands of seconds and blow up the reward.

        Returns shape (K, M) in bps.
        """
        raw = self.bandwidth * np.log2(1.0 + self.SNR)             # (K, M)  bps
        return np.maximum(raw, 1e6)                                 # floor: 1 Mbps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> np.ndarray:
        """
        Regenerate all fast-fading channel values for one time slot.

        Fast-fading (H) changes every slot; path loss, bandwidth, and
        transmit power are treated as slow-varying (fixed per episode).

        Returns
        -------
        rate : np.ndarray, shape (K, M)
            Shannon capacity matrix in bits/second.
            rate[k, m] is the max uplink rate from device k to edge node m.
        """
        self.H    = self._rayleigh_gain()    # Rayleigh envelope,  (K, M)
        self.PL   = self._path_loss()        # Path loss,          (K, M)
        self.SNR  = self._snr()              # Linear SNR,         (K, M)
        self.rate = self._shannon_rate()     # Capacity bps,       (K, M)
        return self.rate

    def get_transmission_delay(self, data_bits: np.ndarray) -> np.ndarray:
        """
        Compute uplink transmission delay for each (device, edge) pair.

            t_tx[k,m] = data_bits[k] / R[k,m]   (seconds)

        Parameters
        ----------
        data_bits : np.ndarray, shape (K,)
            Task data size in bits for each device.

        Returns
        -------
        t_tx : np.ndarray, shape (K, M)  in seconds.
        """
        # data_bits is (K,) → broadcast to (K, M)
        return data_bits[:, np.newaxis] / self.rate                 # (K, M)  seconds
