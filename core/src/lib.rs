//! `contingency_core` extension module for contingency-py.
//!
//! The module itself (`#[pymodule] contingency_core`) is defined in
//! contingency-rs (`contingency::python`). This crate only links it into a
//! `cdylib` so maturin can package it as a wheel; the `PyInit_contingency_core`
//! symbol is exported from the dependency.

pub use contingency::python;
