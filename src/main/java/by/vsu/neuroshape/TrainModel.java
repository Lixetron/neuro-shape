package by.vsu.neuroshape;

import by.vsu.neuroshape.service.ShapeClassifierService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TrainModel {
    private static final Logger log = LoggerFactory.getLogger(TrainModel.class);

    public static void main(String[] args) {
        try {
            log.info("Запуск обучения...");
            ShapeClassifierService service = new ShapeClassifierService();
            service.trainModel();  // Вызов метода обучения
            log.info("Обучение завершено!");
        } catch (Exception e) {
            log.error("Ошибка при обучении: ", e);
        }
    }
}
