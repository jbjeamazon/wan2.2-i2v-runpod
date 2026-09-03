# Keeping this endpoint private

This deployment is meant for one person and their own agent. Nothing here is
protected by obscurity — each item below is a place where access is actually
granted or denied.

## 1. The repository (action required)

`jbjeamazon/wan2.2-i2v-runpod` is currently a **public fork** of
`mindoorio-hue/wan2.2-i2v-runpod`. Two consequences:

- **A fork's visibility cannot be changed.** Visibility is a property of the
  fork network, so there is no "make private" toggle on this repo. See
  [About permissions and visibility of forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-permissions-and-visibility-of-forks).
- **Commits pushed here are served from the parent repo's URL space.** A commit
  pushed to this fork is fetchable at
  `raw.githubusercontent.com/mindoorio-hue/wan2.2-i2v-runpod/<sha>/<path>`.
  Deleting the fork or making it private later does not retract commits that
  are already in the network.

The fix is to **duplicate into a fresh private repository** rather than convert
this one:

```bash
git clone --bare https://github.com/jbjeamazon/wan2.2-i2v-runpod.git
cd wan2.2-i2v-runpod.git
git push --mirror https://github.com/<you>/<new-private-repo>.git
cd .. && rm -rf wan2.2-i2v-runpod.git
```

Then delete the public fork. Treat anything already pushed to it as public.

## 2. The RunPod Hub files

`.runpod/hub.json` and `.runpod/tests.json` exist to list this worker on the
**public RunPod Hub marketplace**, which requires a public repo. They are inert
unless you submit the repo to the Hub — but they are the wrong shape for a
private deployment. For a private endpoint, deploy from a container image or a
private repo connection instead, and delete these two files.

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

## 7. Container images

The GitHub Actions workflows push images to GHCR. At the time of writing these
packages are not anonymously pullable, but **package visibility on GHCR is
independent of repository visibility** and is not changed by making a repo
private. Confirm it directly under the account's Packages settings, and never
bake credentials or LoRAs into a published layer.
