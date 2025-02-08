export interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  content: string[];
  author: string;
  authorAvatar: string;
  date: string;
  imageUrl: string;
  readTime: number;
  category: string;
}