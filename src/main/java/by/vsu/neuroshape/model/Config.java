package by.vsu.neuroshape.model;

import java.util.Arrays;
import java.util.List;

public class Config {
    public static final int HEIGHT = 200;
    public static final int WIDTH = 200;
    public static final int TARGET_HEIGHT = 128;
    public static final int TARGET_WIDTH = 128;
    public static final int CHANNELS = 1; // 3 - RGB, 1 - Gray only

    public static final int BATCH_SIZE = 64;
    public static final int EPOCHS = 15;

    public static final long SEED = 123;

    public static final List<String> LABEL_NAMES = Arrays.asList("circle", "square", "triangle");

    public static final double LEARNING_RATE = 1e-3;
    public static final double L2_REGULARIZATION = 1e-4;

    private Config() {
        throw new IllegalStateException("Utility class");
    }
}
