export interface Post {
  frontmatter: {
    title: string;
    description: string;
    date: string;
    lastUpdated?: string;
    number: string;
    status: string;
    tags: string[];
  };
  url: string;
}

export const posts = Object.values(
  import.meta.glob<Post>('../pages/log/*.md', { eager: true }),
).sort((a, b) => b.frontmatter.date.localeCompare(a.frontmatter.date));

export function formatDate(date: string) {
  return new Intl.DateTimeFormat('en', {
    month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC',
  }).format(new Date(`${date}T12:00:00Z`));
}
