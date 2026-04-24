# Ego-Exo4D Access and Download Guide

Ego-Exo4D is a Meta AI research dataset that requires a formal access request and license agreement. This guide covers the full process from requesting access through downloading the specific subsets used in this project.

---

## Important: Ego-Exo4D vs Ego4D Access

Ego-Exo4D and the original Ego4D dataset are **separate datasets with separate access grants.** Having approval for one does not automatically grant access to the other. Make sure you request Ego-Exo4D specifically, even if you already have Ego4D credentials.

Additionally, the CLI commands are different:
- Ego4D: `ego4d` command
- Ego-Exo4D: `egoexo` command (same `ego4d` pip package, different entry point)

---

## Step 1: Request Access

Go to [ego4d.dev/request/ego4d](https://ego4d.dev/request/ego4d) and complete the access request form. On the form, select Ego-Exo4D (not just the base Ego4D dataset).

You will receive an email with AWS credentials (an Access Key ID and Secret Access Key) once your request is approved. Processing typically takes a few days.

---

## Step 2: Install Dependencies

```bash
pip install ego4d awscli
```

The `ego4d` package installs both the `ego4d` and `egoexo` CLI commands.

---

## Step 3: Configure AWS Credentials

Use the credentials from your approval email:

```bash
aws configure --profile ego4d
```

When prompted:
- AWS Access Key ID: (from your email)
- AWS Secret Access Key: (from your email)
- Default region name: (press Enter to skip)
- Default output format: (press Enter to skip)

To verify your credentials work, try listing something from the regular Ego4D bucket:

```bash
aws s3 ls s3://ego4d-consortium-sharing/ --profile ego4d
```

If this returns a listing, your credentials are valid. If Ego-Exo4D access has also been provisioned, you should also be able to reach the Ego-Exo4D bucket.

---

## Step 4: Test the egoexo CLI

```bash
egoexo --help
```

Try listing available takes to confirm access:

```bash
egoexo --list-takes --s3_profile ego4d
```

### Troubleshooting: "could not get manifests for all parts"

This is the most common error and usually means one of two things:

**Scenario A — Credentials not provisioned for Ego-Exo4D**
Your AWS credentials grant access to Ego4D but not Ego-Exo4D. These are provisioned separately on Meta's backend even when using the same access request form. Contact the dataset maintainers at [discuss.ego4d-data.org](https://discuss.ego4d-data.org) and describe the issue. Include that your `ego4d` CLI works fine with the same credentials but `egoexo` fails with "could not get manifests."

**Scenario B — Wrong `--release` flag**
Try removing the `--release` argument entirely and let the CLI use its default:

```bash
egoexo -o ./egoexo4d_raw --parts metadata --s3_profile ego4d
```

**Scenario C — AWS profile name mismatch**
Make sure the profile name you pass to `--s3_profile` exactly matches what you configured with `aws configure --profile`. If you configured it as `ego4d`, pass `--s3_profile ego4d`.

---

## Step 5: Download the Relevant Subset

This project uses only **exocentric video** from the **cooking** and **bike_repair** scenarios. These scenarios were selected because they emphasize hand-object interaction analogous to robot manipulation.

### Metadata first (always do this before video)

```bash
egoexo \
  -o ./egoexo4d_raw \
  --parts metadata \
  --s3_profile ego4d
```

This downloads the take manifests and annotation files (~a few hundred MB) and is required before any video download.

### Exo video for cooking and bike repair

```bash
egoexo \
  -o ./egoexo4d_raw \
  --parts takes \
  --scenarios cooking,bike_repair \
  --views exo \
  --s3_profile ego4d
```

This downloads only the third-person (exo) camera video for those two scenario categories. Egocentric streams are excluded.

### What gets downloaded

The output directory will have the structure:

```
egoexo4d_raw/
├── takes/
│   ├── <take_uid>/
│   │   ├── frame_aligned_videos/
│   │   │   └── <exo_camera_name>.mp4
│   │   └── ...
│   └── ...
└── annotations/
    └── ...
```

Each take is one recording session, typically 2-10 minutes of video from one or more fixed exo cameras. The approximately 280 takes used in this project combine cooking and bike repair exo streams.

---

## Space Requirements

| Content | Approx size |
|---|---|
| Metadata only | ~500 MB |
| Cooking exo video | ~50-80 GB |
| Bike repair exo video | ~30-60 GB |
| Both (as used in this project) | ~80-150 GB |

Actual sizes depend on resolution and number of takes available at the time of download.

---

## Notes on the Download CLI

The `egoexo` CLI does not support `--max_takes` or partial downloads by count in all versions. If you need to limit the download (e.g., for a quick test), download metadata first, then manually select a subset of take UIDs and use the AWS CLI directly to copy specific files:

```bash
# List takes in a specific scenario from the manifest
# Then copy individual take directories:
aws s3 cp s3://<bucket>/<take_uid>/ ./egoexo4d_raw/takes/<take_uid>/ \
  --recursive --profile ego4d
```

The exact S3 bucket path will be visible in the manifest files after the metadata download. See [ego-exo4d-data.org](https://ego-exo4d-data.org) for updated CLI documentation, as the tooling is under active development.