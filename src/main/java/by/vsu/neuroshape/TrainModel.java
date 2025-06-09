package by.vsu.neuroshape;

import by.vsu.neuroshape.service.ShapeClassifierService;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TrainModel {
    private static final Logger log = LoggerFactory.getLogger(TrainModel.class);

    public static void main(String[] args) {
        try {
            log.info("CUDA доступна: {}", Nd4j.getBackend().isAvailable());
            log.info("Запуск обучения...");
            ShapeClassifierService service = new ShapeClassifierService();
            service.trainModel();  // Вызов метода обучения
            log.info("Обучение завершено!");
            System.exit(0);
        } catch (Exception e) {
            log.error("Ошибка при обучении: ", e);
        }
    }
}
