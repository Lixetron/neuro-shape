package by.vsu.neuroshape.data;

import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.net.URI;
import java.util.List;

public class LabelGenerator implements PathLabelGenerator {
    private final List<String> labelNames;

    public LabelGenerator(List<String> labelNames) {
        this.labelNames = labelNames;
    }

    @Override
    public Writable getLabelForPath(String path) {
        String label = new File(path).getParentFile().getName();
        int idx = labelNames.indexOf(label.toLowerCase());

        if (idx == -1) {
            throw new IllegalArgumentException("Unknown label: " + label);
        }

        return new IntWritable(idx);
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return getLabelForPath(new File(uri).getAbsolutePath());
    }

    @Override
    public boolean inferLabelClasses() {
        return false;
    }
}
