# Keeping this endpoint private

This deployment is meant for one person and their own agent. Nothing here is
protected by obscurity — each item below is a place where access is actually
granted or denied.

## 1. The repository (action required)

The original repo, `jbjeamazon/wan2.2-i2v-runpod`, is a **public fork** of
`mindoorio-hue/wan2.2-i2v-runpod`. Two things follow:

- **A fork's visibility cannot be changed.** Visibility is a property of the
  fork network, so there is no "make private" toggle on it. See
  [About permissions and visibility of forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-permissions-and-visibility-of-forks).
- **Commits pushed to it are served from the parent repo's URL space.** A
  commit pushed to the fork is fetchable at
  `raw.githubusercontent.com/mindoorio-hue/wan2.2-i2v-runpod/<sha>/<path>`.
  Deleting the fork or making it private does not retract commits already in
  the network.

This repository is the replacement: a standalone private repo, not a fork,
seeded with a single commit and no prior history.

**Remaining step: delete the public fork.** Go to
`github.com/jbjeamazon/wan2.2-i2v-runpod` → Settings → Danger Zone → Delete
this repository. Treat everything ever pushed to it as public — if any of it
was sensitive, rotate it rather than relying on the deletion.

## 2. The RunPod Hub files

**Removed.** `.runpod/hub.json` and `.runpod/tests.json` existed to list this
worker on the **public RunPod Hub marketplace**, which requires a public repo.
They have been deleted, along with the README's Hub badge.

Deploy the endpoint from a container image or a private repo connection
instead. `icon.png` is now unreferenced and can go too if you want it gone.
Do not re-add the Hub files unless you intend a public release.

## 3. Endpoint access

The RunPod endpoint is gated by your account's API key. The endpoint ID is not
a secret and should not be treated as one — the key is the only control.

- Issue a **dedicated API key** used only by the MCP server, so it can be
  rotated without disturbing anything else.
- Keep it in the MCP server's environment. Never commit it, and never place it
  in an OpenClaw skill file.
- Rotate it if it ever reaches a chat transcript or a log.

## 4. Generated videos

The videos are the sensitive artifact, not the code.

- Block public access on the bucket. Presigned URLs keep working.
- Leave `S3_PUBLIC_URL_BASE` **unset**. It returns permanent, world-readable
  URLs; it exists for shareable output, which is the opposite of this goal.
- `S3_URL_EXPIRY` defaults to one hour. A presigned URL is an unauthenticated
  bearer link — whoever holds it can fetch the video until it expires, so keep
  the window short and avoid pasting these links into chat history you keep.
- Set a lifecycle rule to delete objects after N days if you do not need them
  retained.

## 5. OpenClaw reachability (the likeliest leak)

OpenClaw bridges chat platforms — WhatsApp, Telegram, Discord, Signal and
others. **Anyone who can message the agent can invoke any tool the agent has**,
including `generate_video`. The endpoint's API key does not help here: the
agent is already authenticated on their behalf.

- Restrict the agent to your own contact/user ID. Do not expose it in group
  chats or shared channels.
- Confirm the restriction empirically by messaging it from a second account.
- Community OpenClaw skills have a documented history of prompt injection and
  credential exfiltration; audit anything you install alongside this, since a
  malicious skill in the same agent can read the RunPod key from the
  environment.

## 6. What RunPod can see

Renting a GPU is not the same as running locally. Input images, prompts and
output videos pass through RunPod's infrastructure, and their Terms of Service
govern what may be generated there. Verify the current terms against your
intended use — this is a policy question, not one the code can settle.

**This is what `local/` exists to remove.** Running on your own GPU eliminates
the operator, the terms of service, and the third-party record in one step.
Nothing in items 3, 4, 6 or 7 applies to a local deployment.

If you stay on rented hardware, note that *serverless* is the worse posture: it
routes every prompt through RunPod's managed queue and dashboard, where it sits
as a queryable job record. Renting a plain GPU pod and running `local/server.py`
on it keeps the payload out of that control plane. The host still owns the
machine — only confidential computing (an H100/H200 in CC mode with an attested
CPU TEE) changes that — but there is no managed queue and no job history.

## 7. Container images

The GitHub Actions workflows push images to GHCR. At the time of writing these
packages are not anonymously pullable, but **package visibility on GHCR is
independent of repository visibility** and is not changed by making a repo
private. Confirm it directly under the account's Packages settings, and never
bake credentials or LoRAs into a published layer.
