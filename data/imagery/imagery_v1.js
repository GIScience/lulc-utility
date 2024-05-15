//VERSION=3

function setup() {
    return {
        input: [
            {
                datasource: "s1",
                bands: ["VV", "VH"],
                processing: {
                    orthorectify: "true"
                }
            },
            {
                datasource: "s2",
                bands: ["B02", "B03", "B04", "B08", "B11", "B12"]
            },
            {
                datasource: "dem",
                bands: ["DEM"]
            }
        ],
        output: [{
            id: "s2",
            bands: 6,
            resx: 10,
            resy: 10,
            sampleType: "UINT8"
        }, {
            id: "s1",
            bands: 2,
            resx: 10,
            resy: 10,
            sampleType: "FLOAT32"
        }, {
            id: "dem",
            bands: 1,
            resx: 10,
            resy: 10,
            sampleType: "FLOAT32"
        }]
    }
}

function evaluatePixel(samples) {
    let s1_sample = samples.s1[0]
    let s2_sample = samples.s2[0]
    let dem_sample = samples.dem[0]

    return {
        "s2": [
            s2_sample.B04 * 255,
            s2_sample.B03 * 255,
            s2_sample.B02 * 255,
            s2_sample.B08 * 255,
            s2_sample.B11 * 255,
            s2_sample.B12 * 255
        ],
        "s1": [
            s1_sample.VV,
            s1_sample.VH
        ],
        "dem": [
            dem_sample.DEM
        ]
    }
}
