# 📬 Enron Email Dataset — Raw Data Directory

## Download

The Enron email dataset is hosted by Carnegie Mellon University:

```bash
# Download (~1.7 GB compressed)
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz

# Or via browser:
# https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
```

## Extraction

Extract the tar.gz archive into this directory:

```bash
# Linux / macOS
tar -xzf enron_mail_20150507.tar.gz -C data/raw/

# Windows (PowerShell)
tar -xzf enron_mail_20150507.tar.gz -C data\raw\
```

## Expected Structure After Extraction

```
data/raw/
└── maildir/
    ├── allen-p/
    │   ├── inbox/
    │   │   ├── 1.
    │   │   ├── 2.
    │   │   └── ...
    │   ├── sent/
    │   ├── sent_items/
    │   ├── _sent_mail/
    │   ├── discussion_threads/
    │   └── ...
    ├── bass-e/
    ├── beck-s/
    ├── lay-k/
    └── ... (~150 employee directories)
```

- **~500,000 emails** across ~150 Enron employees
- Each employee has subdirectories for their email folders
- Individual emails are plain text files with RFC 2822 headers
- **Key folders for training**: `sent/`, `sent_items/`, `_sent_mail/` (authored emails)

## Next Step

After extraction, run the data preparation script:

```bash
python data/prepare_data.py --maildir_path data/raw/maildir
```

This will parse, clean, and format the emails into `train.json` and `val.json`.
