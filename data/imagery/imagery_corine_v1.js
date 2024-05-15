function setup() {
  return {
    input: ["CLC"],
    output: {
      bands: 1,
      resx: 100,
      resy: 100,
      sampleType: "UINT8"
    }
  }
}

function evaluatePixel(sample) {
  return [sample.CLC];
}
