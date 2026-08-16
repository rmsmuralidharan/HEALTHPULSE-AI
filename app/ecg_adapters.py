"""
ECG Input Adapter layer.

Decouples "how do we get a 1000x12 standardized ECG signal into the CNN"
from "what raw format did the hospital/device give us".

              ECG Input Adapter
                     |
       ┌─────────────┼─────────────┐
       |             |             |
     WFDB           XML          DICOM
       |             |             |
       └─────────────┼─────────────┘
                     |
              Standard ECG (1000, 12)
                     |
                    CNN

Only WFDBAdapter is implemented today (matches PTB-XL / our validated
pipeline). XMLAdapter and DICOMAdapter are stubs describing the contract
a future implementation must satisfy, so the UI, prediction, and history
code never need to change when they're added.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict

import numpy as np


# ==========================================================
# Standard ECG container
# ==========================================================

@dataclass
class StandardECG:
    """Model-ready ECG signal: (1000 timesteps, 12 leads).

    Guaranteed: the signal has been through the project's validated
    `DataPreprocessing.preprocess_single_ecg()` pipeline, which includes
    normalization (confirmed). Any additional steps (filtering,
    resampling, etc.) are whatever that pipeline actually implements —
    this docstring intentionally doesn't assert transformations beyond
    normalization until they're confirmed against the pipeline code.
    Adapters are responsible for getting their source format into this
    same (1000, 12), normalized shape before returning a StandardECG.
    """

    signal: np.ndarray   # shape (1000, 12)
    source_format: str   # "wfdb" | "xml" | "dicom" | ...
    source_id: str        # record name / file name, for display + history

    def __post_init__(self):
        if self.signal.ndim != 2:
            raise ValueError(
                f"StandardECG.signal must be 2D, got shape {self.signal.shape}"
            )
        if self.signal.shape != (1000, 12):
            raise ValueError(
                f"StandardECG.signal must be (1000, 12), got {self.signal.shape}. "
                "Adapters are responsible for resampling/lead-mapping to this shape."
            )


# ==========================================================
# Adapter interface
# ==========================================================

class ECGInputAdapter(abc.ABC):
    """Base class for all ECG input adapters.

    An adapter's only job: given source-format-specific input (uploaded
    files, bytes, a hospital-system payload, etc.), produce a
    StandardECG. Everything downstream (CNN, thresholding, UI) only ever
    talks to StandardECG — never to WFDB/XML/DICOM specifics.
    """

    format_name: str = "unknown"

    # User-facing label — plain language, no file-extension jargon.
    display_name: str = "Unknown"

    # Short caption shown under the format picker for this adapter.
    help_text: str = ""

    # Technical name (e.g. actual file types), shown only in
    # dev/technical-details contexts, not as the primary label.
    technical_name: str = ""

    is_implemented: bool = False

    @abc.abstractmethod
    def load(self, **kwargs) -> StandardECG:
        """Load + standardize an ECG from this adapter's source format."""
        raise NotImplementedError


# ==========================================================
# WFDB adapter (implemented — wraps the existing validated pipeline)
# ==========================================================

class WFDBAdapter(ECGInputAdapter):
    """PTB-XL / WFDB (.hea + .dat) adapter.

    Wraps the project's existing, validated `DataPreprocessing`
    component unchanged — same processing as before. This class just
    gives it a common interface. This is the ONLY entry point into
    preprocessing + the CNN for WFDB data — no other code path should
    call `DataPreprocessing` directly.
    """

    format_name = "wfdb"

    # Plain-language label for end users.
    display_name = "WFDB ECG record — upload .hea + .dat"
    help_text = (
        "Currently supports PTB-XL/WFDB ECG recordings. XML and DICOM "
        "support will be added in future versions."
    )
    technical_name = "PTB-XL / WFDB (.hea + .dat)"

    is_implemented = True

    def __init__(self, data_preprocessing_cls, normalization_path: str):
        self._data_preprocessing_cls = data_preprocessing_cls
        self._normalization_path = normalization_path

    def load(self, *, record_dir: str, record_base: str) -> StandardECG:
        preprocessor = self._data_preprocessing_cls(
            data_path=record_dir,
            normalization_path=self._normalization_path,
        )
        processed = preprocessor.preprocess_single_ecg(record_base)
        signal = np.asarray(processed)

        # Normalize orientation to (1000, 12) if the pipeline returns
        # (12, 1000) instead.
        if signal.shape[0] != 1000 and signal.shape[-1] == 1000:
            signal = signal.T

        return StandardECG(
            signal=signal,
            source_format=self.format_name,
            source_id=record_base,
        )


# ==========================================================
# Future adapters (stubs — define the contract, not yet implemented)
# ==========================================================

class XMLAdapter(ECGInputAdapter):
    """Vendor XML ECG exports (e.g. GE MUSE, Philips TraceMasterVue).
    Not implemented yet.

    Contract for a future implementation:
      - Parse per-lead waveforms + sampling rate out of the XML.
      - Map vendor lead order to our canonical 12-lead order.
      - Resample to 1000 samples/lead.
      - Apply whatever normalization/filtering steps
        `DataPreprocessing.preprocess_single_ecg()` is confirmed to
        perform, via a shared, format-agnostic
        `standardize(raw_signal, sampling_rate)` helper so adapters
        don't duplicate or drift from that logic.
    """

    format_name = "xml"
    display_name = "Vendor XML export"
    help_text = "Not available yet — planned for a future version."
    technical_name = "Vendor XML ECG export"
    is_implemented = False

    def load(self, **kwargs) -> StandardECG:
        raise NotImplementedError(
            "XML ECG import is not implemented yet. Planned once "
            "hospital format samples are available."
        )


class DICOMAdapter(ECGInputAdapter):
    """DICOM waveform ECGs (SOP Class: 12-lead ECG Waveform Storage).
    Not implemented yet.

    Contract for a future implementation:
      - Read the DICOM waveform sequence (pydicom or similar).
      - Extract per-lead samples + sampling frequency.
      - Map DICOM lead codes to our canonical 12-lead order.
      - Resample to 1000 samples/lead, then reuse the same shared
        `standardize()` step as XMLAdapter.
    """

    format_name = "dicom"
    display_name = "DICOM waveform"
    help_text = "Not available yet — planned for a future version."
    technical_name = "DICOM 12-lead ECG Waveform Storage"
    is_implemented = False

    def load(self, **kwargs) -> StandardECG:
        raise NotImplementedError(
            "DICOM ECG import is not implemented yet. Planned once "
            "hospital format samples are available."
        )


# ==========================================================
# Registry / factory
# ==========================================================

def build_adapter_registry(
    data_preprocessing_cls, normalization_path: str
) -> Dict[str, ECGInputAdapter]:
    """Construct all known adapters, keyed by format_name.

    Only WFDB is functional today; XML/DICOM are registered so the UI
    can list them as "coming soon" without special-casing the list
    elsewhere in the app. WFDB is listed first and should be treated
    as the default selection in the UI.
    """
    return {
        "wfdb": WFDBAdapter(data_preprocessing_cls, normalization_path),
        "xml": XMLAdapter(),
        "dicom": DICOMAdapter(),
    }