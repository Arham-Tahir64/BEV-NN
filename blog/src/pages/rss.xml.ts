import type { APIRoute } from 'astro';
import { posts } from '../lib/posts';

const escape = (value: string) => value.replace(/[<>&"']/g, character => ({
  '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;',
})[character]!);

export const GET: APIRoute = ({ site }) => {
  const home = new URL(`${import.meta.env.BASE_URL.replace(/\/$/, '')}/`, site);
  const items = posts.map(({ frontmatter: post, url }) => {
    const link = new URL(url, site).href;
    return `<item><title>${escape(post.title)}</title><description>${escape(post.description)}</description><link>${escape(link)}</link><guid isPermaLink="true">${escape(link)}</guid><pubDate>${new Date(`${post.date}T12:00:00Z`).toUTCString()}</pubDate></item>`;
  }).join('');
  return new Response(`<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>BEV-NN — field notes</title><link>${escape(home.href)}</link><description>Arham Tahir’s camera-only 3D occupancy development log.</description><language>en</language>${items}</channel></rss>`, { headers: { 'Content-Type': 'application/rss+xml; charset=utf-8' } });
};
