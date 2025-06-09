package by.vsu.neuroshape.service;

import by.vsu.neuroshape.data.ImageDataLoader;
import by.vsu.neuroshape.model.ModelConfig;
import by.vsu.neuroshape.model.NeuralNetwork;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransformProcess;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Objects;

import static by.vsu.neuroshape.model.Config.*;

public class ShapeClassifierService {
    private static final Logger log = LoggerFactory.getLogger(ShapeClassifierService.class);

    private static final String MODEL_PATH = "shape-model.zip";

    // Пути к данным (ваша структура)
    private static final String TRAIN_DATA_PATH = "src/main/resources/shapes/train";
    private static final String TEST_DATA_PATH = "src/main/resources/shapes/test";

    private NeuralNetwork model;
    private UIServer uiServer;

    // Обучение модели
    public void trainModel() throws Exception {
        log.info("Проверка данных в {}", TRAIN_DATA_PATH);
        File trainDir = new File(TRAIN_DATA_PATH);

        if (!trainDir.exists() || Objects.requireNonNull(trainDir.list()).length == 0) {
            throw new FileNotFoundException("Данные для обучения не найдены или отсутствуют");
        }

        LABEL_NAMES.forEach(label -> {
            File dir = new File(trainDir, label);
            int count = dir.exists()
                    ? Objects.requireNonNull(dir.list((d, name) -> name.endsWith(".png") || name.endsWith(".jpg"))).length
                    : 0;
            log.info("Примеров {}: {}", label, count);
        });

        DataSetIterator trainIter = new AsyncDataSetIterator(
                ImageDataLoader.loadTrainData(
                        TRAIN_DATA_PATH,
                        BATCH_SIZE,
                        LABEL_NAMES
                ),
                2);

        // 2. Инициализация UI-сервера
        uiServer = UIServer.getInstance();
        InMemoryStatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        log.info("Создание модели...");
        model = new NeuralNetwork(
                ModelConfig.getConfig(CHANNELS, LABEL_NAMES.size()),
                new ScoreIterationListener(10),
                new StatsListener(statsStorage)
                //new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END)
        );

        log.info("Обучение модели ({} эпох)...", EPOCHS);
        model.train(trainIter, EPOCHS);

        log.info("Сохранение модели в {}...", MODEL_PATH);
        ModelSerializer.writeModel(model.getModel(), new File(MODEL_PATH), true);
    }

    /**
     * Предсказывает фигуру на изображении
     *
     * @param imagePath путь к изображению (относительно resources)
     *
     * @return название фигуры ("circle", "square" или "triangle")
     */
    public String predictShape(String imagePath) throws Exception {
        if (model == null) {
            loadModelIfExists();
        }

        log.debug("Загрузка тестового изображения: {}...", imagePath);
        NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
        ImageWritable imageWritable = loader.asWritable(new File(imagePath));

        ImageTransformProcess transformProcess = new ImageTransformProcess.Builder()
                .resizeImageTransform(TARGET_WIDTH, TARGET_HEIGHT)
                .build();

        INDArray imageArray = loader.asMatrix(transformProcess.execute(imageWritable));

        log.debug("Нормализация изображения...");
        new ImagePreProcessingScaler(0, 1).transform(imageArray);

        log.debug("Выполнение предсказания...");
        INDArray output = model.predict(imageArray);
        int predicted = output.argMax(1).getInt(0);

        return LABEL_NAMES.get(predicted);
    }

    /**
     * Пакетное тестирование на всех изображениях из test/
     */
    public void testOnAllImages() throws Exception {
        if (model == null) {
            loadModelIfExists();
        }

        int correct = 0;
        int total = 0;

        for (String shape : LABEL_NAMES) {
            File[] files = new File(TEST_DATA_PATH, shape)
                    .listFiles((dir, name) -> name.endsWith(".png") || name.endsWith(".jpg"));

            if (files == null) {
                continue;
            }

            log.info("Тестирование {} изображений {}", files.length, shape);
            for (File file : files) {
                String predicted = predictShape(file.getAbsolutePath());

                if (predicted.equals(shape)) {
                    correct++;
                } else {
                    log.warn("Ошибка: {} → {}", file.getName(), predicted);
                }

                total++;
            }
        }

        double accuracy = 100.0 * correct / total;

        log.info("Точность: {}/{} ({:.2f}%)", correct, total, accuracy);
    }

    /**
     * Загружает модель из файла (если она существует)
     */
    private void loadModelIfExists() throws Exception {
        if (new File(MODEL_PATH).exists()) {
            log.info("Загрузка модели из {}...", MODEL_PATH);
            model = new NeuralNetwork(ModelSerializer.restoreMultiLayerNetwork(MODEL_PATH));
        } else {
            throw new IllegalStateException("Модель не найдена. Сначала выполните обучение.");
        }
    }
}
