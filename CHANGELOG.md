# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project mostly adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://gitlab.heigit.org/climate-action/utilities/lulc-utility/-/compare/1.1.1...main)

### Fixed

- initialise log level early and customise log config within `app.logger`

### Changed

- environment and dependency management from `conda` to [`uv`](https://docs.astral.sh/uv/)
- area descriptor is now file-based or using OSM boundaries instead of NUTS area
  definitions ([#59](https://gitlab.heigit.org/climate-action/utilities/lulc-utility/-/issues/59))

### Added

- a debug message explaining that the input AOI is buffered and how big the new bounding box is (addresses confusion
  in [#80](https://gitlab.heigit.org/climate-action/utilities/lulc-utility/-/issues/80))

## [1.1.1](https://gitlab.heigit.org/climate-action/utilities/lulc-utility/-/releases/1.1.1) - 2025-06-02

### Fixed

- memory leak in the api inference ([#82](https://gitlab.heigit.org/climate-action/utilities/lulc-utility/-/issues/82))


## [1.1.0](https://gitlab.heigit.org/climate-action/utilities/lulc-utility/-/releases/1.1.0) - 2024-09-06

### Fixed

- the utility returning all-zero data for the /segment endpoint

### Added

- a dedicated endpoint to calculate the uncertainty of the classification
- possibility for CORINE data harmonisation
- improved the sampling strategy for overlapping geometries

## [1.0.1](https://gitlab.heigit.org/climate-action/utilities/lulc-utility/-/releases/1.0.1) - 2024-06-21

### Fix

- remove caching from classification function

## [1.0.0](https://gitlab.heigit.org/climate-action/utilities/lulc-utility/-/releases/1.0.0) - 2024-05-22

### Added

- a stand-alone REST-API based utility to request and preview LULC classifications based on Sentinel-1, 2 and DEM using
  custom trained deep learning models
- scripts and definitions to train custom deep learning models for LULC classification based on Sentiel-1, 2 and DEM
  data using OSM LULC data as labels