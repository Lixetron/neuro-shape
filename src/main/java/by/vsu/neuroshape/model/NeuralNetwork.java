package by.vsu.neuroshape.model;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.stream.IntStream;

public class NeuralNetwork {
    private final MultiLayerNetwork model;

    public NeuralNetwork(MultiLayerConfiguration config,
                         TrainingListener... listeners) {
        this.model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(listeners);
    }

    public NeuralNetwork(MultiLayerNetwork model) {
        this.model = model;
    }

    public void train(DataSetIterator trainIter, int epochs) {
        IntStream.range(0, epochs).forEach(i -> {
            trainIter.reset();
            model.fit(trainIter);
        });
    }

    public void addListeners(TrainingListener... listeners) {
        model.setListeners(listeners);
    }

    public INDArray predict(INDArray input) {
        return model.output(input);
    }

    public MultiLayerNetwork getModel() {
        return model;
    }
}
