import { useParams } from "react-router";
import { posts } from "../data/posts";
import { Clock, Share2, Bookmark } from "lucide-react";

export default function Article() {
  const { id } = useParams<{ id: string }>(); // Extract the article ID from URL params

  const post = posts.find((p) => p.id === id);

  if (!post) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-12">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Article not found
          </h1>
        </div>
      </div>
    );
  }

  return (
    <article className="min-h-screen bg-gray-50 dark:bg-gray-900 py-12">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <span className="px-3 py-1 text-sm font-medium text-indigo-700 dark:text-indigo-400 bg-indigo-100 dark:bg-indigo-900/30 rounded-full">
              {post.category}
            </span>
            <div className="flex items-center space-x-4">
              <button className="p-2 text-gray-600 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800">
                <Share2 className="h-5 w-5" />
              </button>
              <button className="p-2 text-gray-600 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800">
                <Bookmark className="h-5 w-5" />
              </button>
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            {post.title}
          </h1>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <img
                src={post.authorAvatar}
                alt={post.author}
                className="w-12 h-12 rounded-full object-cover border-2 border-white dark:border-gray-800"
              />
              <div>
                <span className="block font-medium text-gray-900 dark:text-white">
                  {post.author}
                </span>
                <span className="block text-sm text-gray-500 dark:text-gray-400">
                  {post.date}
                </span>
              </div>
            </div>
            <div className="flex items-center text-gray-600 dark:text-gray-400">
              <Clock className="h-5 w-5 mr-2" />
              <span>{post.readTime} min read</span>
            </div>
          </div>
        </header>

        {/* Featured Image */}
        <div className="relative h-[400px] rounded-2xl overflow-hidden mb-12">
          <img
            src={post.imageUrl}
            alt={post.title}
            className="absolute inset-0 w-full h-full object-cover"
          />
        </div>

        {/* Article Content */}
        <div className="prose dark:prose-invert max-w-none">
          {post.content.map((paragraph: string, index: number) => (
            <div key={index} dangerouslySetInnerHTML={{ __html: paragraph }} />
          ))}
        </div>
      </div>
    </article>
  );
}
