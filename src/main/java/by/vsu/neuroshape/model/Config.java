package by.vsu.neuroshape.model;

import java.util.Arrays;
import java.util.List;

public class Config {
    public static final int HEIGHT = 56;
    public static final int WIDTH = 56;
    public static final int CHANNELS = 3; // 3 - RGB, 1 - Gray only

    public static final int BATCH_SIZE = 32;
    public static final int EPOCHS = 50;

    public static final long SEED = 123;

    public static final List<String> LABEL_NAMES = Arrays.asList("circle", "square", "triangle");

    public static final double LEARNING_RATE = 1e-3;
    public static final double L2_REGULARIZATION = 1e-4;
}
