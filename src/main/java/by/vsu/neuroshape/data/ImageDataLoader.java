package by.vsu.neuroshape.data;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.Random;

public class ImageDataLoader {
    private static final int HEIGHT = 64;
    private static final int WIDTH = 64;
    private static final int CHANNELS = 3; // 3 канала для RGB

    public static DataSetIterator loadTrainData(String trainDataPath, int batchSize, List<String> labelNames) throws Exception {
        File trainData = new File(trainDataPath);

        if (!trainData.exists()) {
            throw new FileNotFoundException("Директория для обучения не найдена: " + trainDataPath);
        }

        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random(123));

        // Извлекаем метки из названия папки
        ImageRecordReader trainReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, new LabelGenerator(labelNames));
        trainReader.initialize(trainSplit);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, batchSize, 1, labelNames.size());
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

        return trainIter;
    }

    private ImageDataLoader() {
        throw new IllegalStateException("Utility class");
    }
}
