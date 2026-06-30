package com.mcbowagent.record;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import net.minecraft.client.texture.NativeImage;

/**
 * Off-load PNG encode + disk write from the render thread to a background daemon.
 * The render thread builds the downscaled NativeImage and hands it to {@link #submit};
 * a worker thread drains the queue, calls {@code NativeImage.writeFile} + closes it.
 *
 * The queue is small (capacity 4) so back-pressure surfaces as DROPS rather than
 * unbounded memory growth — at 10 Hz capture and reasonable disk speed the queue
 * stays at 0-1 items; if disk stalls (other process hammering it), older frames are
 * dropped instead of blocking the game.
 */
public final class AsyncFrameWriter {

    private static final Logger LOGGER = LogManager.getLogger("mcbowagent");
    private static final int QUEUE_CAPACITY = 4;

    private static final class Job {
        final NativeImage image;
        final Path outFile;
        Job(NativeImage image, Path outFile) { this.image = image; this.outFile = outFile; }
    }

    private final BlockingQueue<Job> queue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private final Thread worker;
    private volatile boolean running = true;
    private final AtomicLong dropped = new AtomicLong();
    private final AtomicLong written = new AtomicLong();

    public AsyncFrameWriter() {
        worker = new Thread(this::loop, "mcbowagent-png-writer");
        worker.setDaemon(true);
        worker.start();
    }

    /** Try to enqueue a frame for async writing. Returns true on success, false if the queue is full
     *  (in which case the caller closes the image and the frame is dropped). The image must be CLOSED
     *  by whoever ends up with it — successful submit transfers ownership to the worker; on failure the
     *  caller is responsible. */
    public boolean submit(NativeImage img, Path outFile) {
        if (!running) return false;
        Job j = new Job(img, outFile);
        if (queue.offer(j)) return true;
        dropped.incrementAndGet();
        return false;
    }

    public long droppedCount() { return dropped.get(); }
    public long writtenCount() { return written.get(); }

    public void close() {
        running = false;
        worker.interrupt();
        try { worker.join(2000); } catch (InterruptedException ignored) {}
        // drain anything left, closing the NativeImages so we don't leak
        Job j;
        while ((j = queue.poll()) != null) {
            try { j.image.close(); } catch (Exception ignored) {}
        }
    }

    private void loop() {
        while (running) {
            Job j;
            try {
                j = queue.take();
            } catch (InterruptedException e) {
                if (!running) return;
                continue;
            }
            try {
                if (j.outFile.getParent() != null) {
                    Files.createDirectories(j.outFile.getParent());
                }
                j.image.writeFile(j.outFile.toFile());
                written.incrementAndGet();
            } catch (IOException | RuntimeException e) {
                LOGGER.warn("[mcbowagent] async PNG write failed for {}: {}", j.outFile, e.toString());
            } finally {
                try { j.image.close(); } catch (Exception ignored) {}
            }
        }
    }
}
