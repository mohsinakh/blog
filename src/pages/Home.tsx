import BlogCard from '../components/BlogCard';
import FeaturedPost from '../components/FeaturedPost';
import NewsletterCTA from '../components/NewsletterCTA';
import { posts } from '../data/posts';

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50 dark:bg-gray-900 pt-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <FeaturedPost post={posts[0]} />
        
        <section className="mb-16">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white">Latest Articles</h2>
            <a href="/articles" className="text-indigo-600 dark:text-indigo-400 font-medium hover:text-indigo-800 dark:hover:text-indigo-300">
              View all articles →
            </a>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {posts.slice(1).map((post) => (
              <BlogCard key={post.id} post={post} />
            ))}
          </div>
        </section>
        
        <NewsletterCTA />
      </div>
    </main>
  );
}