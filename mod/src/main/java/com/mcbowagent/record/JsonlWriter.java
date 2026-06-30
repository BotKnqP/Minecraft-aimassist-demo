package com.mcbowagent.record;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/** Appends one JSON object per line (UTF-8). Flushes each line — fine at 20 Hz. */
public final class JsonlWriter implements Closeable {

    private final BufferedWriter writer;

    public JsonlWriter(Path file) throws IOException {
        if (file.getParent() != null) {
            Files.createDirectories(file.getParent());
        }
        this.writer = Files.newBufferedWriter(file, StandardCharsets.UTF_8);
    }

    public void writeLine(String line) throws IOException {
        writer.write(line);
        writer.write('\n');
        writer.flush();
    }

    @Override
    public void close() throws IOException {
        writer.close();
    }
}
