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

            // Пример предсказания для одного изображения
            String result = service.predictShape("src/main/resources/shapes/test/example.png");
            log.info("Предсказанная фигура: {}", result);
        } catch (Exception e) {
            log.error("Ошибка при тестировании: ", e);
        }
    }
}
