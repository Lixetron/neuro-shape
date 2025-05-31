package by.vsu.neuroshape.data;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.Random;

import static by.vsu.neuroshape.model.Config.*;

public class ImageDataLoader {

    public static DataSetIterator loadTrainData(String trainDataPath, int batchSize, List<String> labelNames) throws Exception {
        File trainData = new File(trainDataPath);

        if (!trainData.exists()) {
            throw new FileNotFoundException("Директория для обучения не найдена: " + trainDataPath);
        }

        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random(123));

        // === 🔄 Аугментации ===
        ImageTransform transform = new PipelineImageTransform(
                //               new FlipImageTransform(1), // горизонтальное отражение
                //               new WarpImageTransform(new Random(123), 10),
                //               new RotateImageTransform(new Random(123), 15),
                //               new ScaleImageTransform(0.9f) // масштаб
        );

        ImageRecordReader trainReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, new LabelGenerator(labelNames));
        trainReader.initialize(trainSplit, transform);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, batchSize, 1, labelNames.size());

        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

        return trainIter;
    }

    private ImageDataLoader() {
        throw new IllegalStateException("Utility class");
    }
}
