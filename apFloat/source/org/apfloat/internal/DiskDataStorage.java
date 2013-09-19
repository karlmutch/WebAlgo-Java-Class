package org.apfloat.internal;

import java.io.Serializable;
import java.io.File;
import java.io.RandomAccessFile;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.lang.ref.SoftReference;

import org.apfloat.ApfloatContext;
import org.apfloat.ApfloatRuntimeException;
import org.apfloat.spi.ArrayAccess;
import org.apfloat.spi.DataStorage;
import org.apfloat.spi.FilenameGenerator;

/**
 * Abstract base class for disk-based data storage, containing the common
 * functionality independent of the element type.
 *
 * @version 1.5.1
 * @author Mikko Tommila
 */

public abstract class DiskDataStorage
    extends DataStorage
{
    private static class FileStorage
        implements Serializable
    {
        public FileStorage()
            throws ApfloatRuntimeException
        {
            init();
        }

        private void init()
            throws ApfloatRuntimeException
        {
            ApfloatContext ctx = ApfloatContext.getContext();
            FilenameGenerator generator = ctx.getFilenameGenerator();

            this.filename = generator.generateFilename();

            this.file = new File(this.filename);

            try
            {
                if (!this.file.createNewFile())
                {
                    throw new BackingStorageException("Failed to create new file \"" + this.filename + '\"');
                }

                // Ensure file is deleted always
                this.file.deleteOnExit();

                this.randomAccessFile = new RandomAccessFile(this.file, "rw");
            }
            catch (IOException ioe)
            {
                throw new BackingStorageException("Unable to access file \"" + this.filename + '\"', ioe);
            }

            this.fileChannel = this.randomAccessFile.getChannel();
        }

        protected void finalize()
        {
            try
            {
                this.fileChannel.close();
            }
            catch (IOException ioe)
            {
                // Ignore
            }

            try
            {
                this.randomAccessFile.close();
            }
            catch (IOException ioe)
            {
                // Ignore
            }

            // If deletion fails now, at least deleteOnExit() has been called
            this.file.delete();
        }

        public void setSize(long size)
            throws IOException
        {
            try
            {
                this.randomAccessFile.setLength(size);
            }
            catch (IOException ioe)
            {
                // Run garbage collection to delete unused temporary files, then retry
                System.gc();
                this.randomAccessFile.setLength(size);
            }
        }

        public void transferFrom(ReadableByteChannel in, long position, long size)
            throws ApfloatRuntimeException
        {
            try
            {
                if (in instanceof FileChannel)
                {
                    // Optimized transferFrom() between two FileChannels
                    while (size > 0)
                    {
                        long count = getFileChannel().transferFrom(in, position, size);
                        position += count;
                        size -= count;
                        assert (size >= 0);
                    }
                }
                else
                {
                    // The FileChannel transferFrom() uses an 8kB buffer, which is too small and inefficient
                    // So we use a similar mechanism but with a custom buffer size
                    ByteBuffer buffer = getDirectByteBuffer();
                    while (size > 0)
                    {
                        buffer.clear();
                        int readCount = (int) Math.min(size, buffer.capacity());
                        buffer.limit(readCount);
                        readCount = in.read(buffer);
                        buffer.flip();
                        while (readCount > 0)
                        {
                            int writeCount = getFileChannel().write(buffer, position);
                            position += writeCount;
                            size -= writeCount;
                            readCount -= writeCount;
                        }
                        assert (readCount == 0);
                        assert (size >= 0);
                    }
                }
            }
            catch (IOException ioe)
            {
                throw new BackingStorageException("Unable to write to file \"" + getFilename() + '\"', ioe);
            }
        }

        public void transferTo(WritableByteChannel out, long position, long size)
            throws ApfloatRuntimeException
        {
            try
            {
                if (out instanceof FileChannel)
                {
                    // Optimized transferTo() between two FileChannels
                    while (size > 0)
                    {
                        long count = getFileChannel().transferTo(position, size, out);
                        position += count;
                        size -= count;
                        assert (size >= 0);
                    }
                }
                else
                {
                    // The DiskChannel transferTo() uses an 8kB buffer, which is too small and inefficient
                    // So we use a similar mechanism but with a custom buffer size
                    ByteBuffer buffer = getDirectByteBuffer();
                    while (size > 0)
                    {
                        buffer.clear();
                        int readCount = (int) Math.min(size, buffer.capacity());
                        buffer.limit(readCount);
                        readCount = getFileChannel().read(buffer, position);
                        buffer.flip();
                        while (readCount > 0)
                        {
                            int writeCount = out.write(buffer);
                            position += writeCount;
                            size -= writeCount;
                            readCount -= writeCount;
                        }
                        assert (readCount == 0);
                        assert (size >= 0);
                    }
                }
            }
            catch (IOException ioe)
            {
                throw new BackingStorageException("Unable to read from file \"" + getFilename() + '\"', ioe);
            }
        }

        public String getFilename()
        {
            return this.filename;
        }

        public FileChannel getFileChannel()
        {
            return this.fileChannel;
        }

        // Writes the file contents to the serialization stream
        private void writeObject(java.io.ObjectOutputStream out)
            throws IOException
        {
            long size = getFileChannel().size();
            out.writeLong(size);

            transferTo(Channels.newChannel(out), 0, size);

            out.defaultWriteObject();
        }

        // Reads file contents from the serialization stream
        private void readObject(java.io.ObjectInputStream in)
            throws IOException, ClassNotFoundException
        {
            init();

            long size = in.readLong();

            setSize(size);

            transferFrom(Channels.newChannel(in), 0, size);

            in.defaultReadObject();
        }

        private static final long serialVersionUID = 2062430603153403341L;

        // These fields are not serialized automatically
        private transient String filename;
        private transient File file;
        private transient RandomAccessFile randomAccessFile;
        private transient FileChannel fileChannel;
    }

    /**
     * Default constructor.
     */

    protected DiskDataStorage()
        throws ApfloatRuntimeException
    {
        this.fileStorage = new FileStorage();
    }

    /**
     * Subsequence constructor.
     *
     * @param diskDataStorage The originating data storage.
     * @param offset The subsequence starting position.
     * @param length The subsequence length.
     */

    protected DiskDataStorage(DiskDataStorage diskDataStorage, long offset, long length)
    {
        super(diskDataStorage, offset, length);
        this.fileStorage = diskDataStorage.fileStorage;
    }

    protected void implCopyFrom(DataStorage dataStorage, long size)
        throws ApfloatRuntimeException
    {
        if (dataStorage == this)
        {
            setSize(size);
            return;
        }

        assert (size > 0);
        assert (!isReadOnly());
        assert (!isSubsequenced());

        int unitSize = getUnitSize();
        long byteSize = size * unitSize;

        assert (byteSize > 0);

        try
        {
            this.fileStorage.setSize(byteSize);

            long readSize = Math.min(size, dataStorage.getSize()),
                 oldSize = readSize * unitSize,
                 padSize = byteSize - oldSize;

            if (dataStorage instanceof DiskDataStorage)
            {
                // Optimized disk-to-disk copy

                DiskDataStorage that = (DiskDataStorage) dataStorage;
                that.transferTo(getFileChannel().position(0),
                                that.getOffset() * unitSize,
                                oldSize);
            }
            else
            {
                // Un-optimized copy from arbitrary data storage

                long position = 0;
                int bufferSize = getBlockSize() / unitSize;
                while (readSize > 0)
                {
                    int length = (int) Math.min(bufferSize, readSize);

                    ArrayAccess readArrayAccess = dataStorage.getArray(READ, position, length);
                    ArrayAccess writeArrayAccess = getArray(WRITE, position, length);
                    System.arraycopy(readArrayAccess.getData(), readArrayAccess.getOffset(), writeArrayAccess.getData(), writeArrayAccess.getOffset(), length);
                    writeArrayAccess.close();
                    readArrayAccess.close();

                    readSize -= length;
                    position += length;
                }
            }
            pad(oldSize, padSize);
        }
        catch (IOException ioe)
        {
            throw new BackingStorageException("Unable to copy to file \"" + getFilename() + '\"', ioe);
        }
    }

    protected long implGetSize()
        throws ApfloatRuntimeException
    {
        try
        {
            return getFileChannel().size() / getUnitSize();
        }
        catch (IOException ioe)
        {
            throw new BackingStorageException("Unable to access file \"" + getFilename() + '\"', ioe);
        }
    }

    protected void implSetSize(long size)
        throws ApfloatRuntimeException
    {
        assert (size > 0);
        assert (!isReadOnly());
        assert (!isSubsequenced());

        size *= getUnitSize();

        assert (size > 0);

        try
        {
            long oldSize = getFileChannel().size(),
                 padSize = size - oldSize;
            this.fileStorage.setSize(size);
            pad(oldSize, padSize);
        }
        catch (IOException ioe)
        {
            throw new BackingStorageException("Unable to access file \"" + getFilename() + '\"', ioe);
        }
    }

    /**
     * Transfer from a readable channel, possibly in multiple chunks.
     *
     * @param in Input channel.
     * @param position Start position of transfer.
     * @param size Total number of bytes to transfer.
     */

    protected void transferFrom(ReadableByteChannel in, long position, long size)
        throws ApfloatRuntimeException
    {
        this.fileStorage.transferFrom(in, position, size);
    }

    /**
     * Transfer to a writable channel, possibly in multiple chunks.
     *
     * @param out Output channel.
     * @param position Start position of transfer.
     * @param size Total number of bytes to transfer.
     */

    protected void transferTo(WritableByteChannel out, long position, long size)
        throws ApfloatRuntimeException
    {
        this.fileStorage.transferTo(out, position, size);
    }

    /**
     * Convenience method for getting the block size (in bytes) for the
     * current {@link ApfloatContext}.
     *
     * @return I/O block size, in bytes.
     */

    protected static int getBlockSize()
    {
        ApfloatContext ctx = ApfloatContext.getContext();
        return ctx.getBlockSize();
    }

    /**
     * Size of the element type, in bytes.
     *
     * @return Size of the element type, in bytes.
     */

    protected abstract int getUnitSize();

    /**
     * Filename of the underlying disk data storage.
     *
     * @return Filename of the underlying disk data storage.
     */

    protected final String getFilename()
    {
        return this.fileStorage.getFilename();
    }

    /**
     * The <code>FileChannel</code> of the underlying disk file.
     *
     * @return The <code>FileChannel</code> of the underlying disk file.
     */

    protected final FileChannel getFileChannel()
    {
        return this.fileStorage.getFileChannel();
    }

    private void pad(long position, long size)
        throws IOException, ApfloatRuntimeException
    {
        transferFrom(ZERO_CHANNEL, position, size);
    }

    private static final ReadableByteChannel ZERO_CHANNEL = new ReadableByteChannel()
    {
        public int read(ByteBuffer buffer)
        {
            int writeLength = buffer.remaining();

            for (int i = 0; i < writeLength; i++)
            {
                buffer.put((byte) 0);
            }

            return writeLength;
        }

        public void close() {}
        public boolean isOpen() { return true; }
    };

    private static ByteBuffer getDirectByteBuffer()
    {
        // Since direct buffers are allocated outside of the heap they can behave strangely in relation to GC
        // So we try to make them as long-lived as possible and cache them in a ThreadLocal
        ByteBuffer buffer = null;
        int blockSize = getBlockSize();
        SoftReference<ByteBuffer> reference = DiskDataStorage.threadLocal.get();
        if (reference != null)
        {
            buffer = reference.get();
            if (buffer != null && buffer.capacity() != blockSize)
            {
                // Clear references to the direct buffer so it may be GC'd
                reference.clear();
                buffer = null;
            }
        }
        if (buffer == null)
        {
            buffer = ByteBuffer.allocateDirect(blockSize);
            reference = new SoftReference<ByteBuffer>(buffer);
            DiskDataStorage.threadLocal.set(reference);
        }

        return buffer;
    }

    private static final long serialVersionUID = 741984828408146034L;

    private static ThreadLocal<SoftReference<ByteBuffer>> threadLocal = new ThreadLocal<SoftReference<ByteBuffer>>();

    private FileStorage fileStorage;
}
