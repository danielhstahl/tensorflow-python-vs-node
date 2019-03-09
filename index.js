const tf = require('@tensorflow/tfjs-node')
const data =require('./external_scripts/data')
const loadData=()=>{
    return data.loadData().then(()=>{
        const train=data.getTrainData()
        const test=data.getTestData()
        return {xTrain:train.images, yTrain:train.labels, xTest:test.images, yTest:test.labels}
    })
}
const IMAGE_H=28
const IMAGE_W=IMAGE_H
const convertTensorToMetric=tensor=>tensor.dataSync()[0].toFixed(3)
const train_model=(xTrain, yTrain)=>{
    const model = tf.sequential()
    model.add(tf.layers.flatten({inputShape: [IMAGE_H, IMAGE_W, 1]}))
    model.add(tf.layers.dense({units: 128, activation: 'relu'}))
    model.add(tf.layers.dropout({rate:0.2}))
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}))
    model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics:['accuracy']})
    return model.fit(xTrain, yTrain, {
        epochs: 5,
        batchSize: 300,
        callbacks: {
            onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        }
    }).then(()=>{
        return model
    })
}


loadData()
.then(({xTrain, yTrain, xTest, yTest})=>{
    return train_model(xTrain, yTrain).then(model=>{
        const eval=model.evaluate(xTest, yTest)
        const accuracy=convertTensorToMetric(eval[1])
        console.log(`Accuracy: ${accuracy}`)
    })
})


