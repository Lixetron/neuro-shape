package by.vsu.neuroshape;

import by.vsu.neuroshape.service.ShapeClassifierService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestModel {
    private static final Logger log = LoggerFactory.getLogger(TestModel.class);

    public static void main(String[] args) {
        try {
            log.info("Запуск тестирования...");
            ShapeClassifierService service = new ShapeClassifierService();

            service.testOnAllImages();
        } catch (Exception e) {
            log.error("Ошибка при тестировании: ", e);
        }
    }
}
