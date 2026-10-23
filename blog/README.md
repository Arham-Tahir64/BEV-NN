# BEV-NN development log

An Astro blog about building a camera-only 3D semantic occupancy model.

Live URL: https://arham-tahir64.github.io/BEV-NN/

## Local development

Use Node.js 24 (or Node.js 22.23.4 or later).

```sh
cd blog
npm ci
npm run dev
```

Open the local URL printed by Astro, including `/BEV-NN/`.

```sh
npm run check
npm run build
npm run preview
```

## Write the next post

Add a Markdown file in `src/pages/log/`, using the first post as a template:

```yaml
---
layout: ../../layouts/Post.astro
title: "Your next development milestone"
description: "A short summary for the archive, search previews, and RSS feed."
date: "2026-10-22"
number: "02"
status: "Implementation"
tags: ["PyTorch", "3D perception"]
---
```

Use `##` headings for sections; the article sidebar is generated from them.
The homepage and RSS feed automatically include posts, newest first.
Commit and push to `main` to publish. The date above is an example, not a scheduled post.

## GitHub Pages

`.github/workflows/deploy-blog.yml` installs locked dependencies, checks the
Astro source, builds `blog/dist`, and deploys it to GitHub Pages. In the repository
settings, Pages → Build and deployment → Source must be **GitHub Actions**.
Once enabled, later blog pushes deploy automatically.

The config uses `site: https://arham-tahir64.github.io` and `base: /BEV-NN`.
Update both when moving to another repository or a custom domain.

Reference: https://docs.astro.build/en/guides/deploy/github/
