import { Clock, ArrowRight, BookmarkPlus } from 'lucide-react';
import { BlogPost } from '../types/blog';

interface FeaturedPostProps {
  post: BlogPost;
}

export default function FeaturedPost({ post }: FeaturedPostProps) {
  return (
    <article className="relative bg-gradient-to-br from-indigo-600 to-purple-700 dark:from-indigo-900 dark:to-purple-900 rounded-2xl shadow-2xl overflow-hidden mb-12">
      <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&q=80')] opacity-10 mix-blend-overlay" />
      <div className="relative flex flex-col lg:flex-row">
        <div className="lg:w-1/2 p-8 lg:p-12">
          <div className="flex items-center justify-between mb-6">
            <span className="px-4 py-1.5 text-sm font-semibold text-white bg-white/20 backdrop-blur-sm rounded-full">
              Featured Post
            </span>
            <button className="p-2 text-white/80 hover:text-white rounded-full hover:bg-white/10 transition-colors">
              <BookmarkPlus className="h-5 w-5" />
            </button>
          </div>
          <h1 className="text-4xl font-bold text-white mb-6 leading-tight">{post.title}</h1>
          <p className="text-lg text-white/90 mb-8">{post.excerpt}</p>
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center space-x-4">
              <img 
                src={post.authorAvatar} 
                alt={post.author}
                className="w-10 h-10 rounded-full border-2 border-white/20"
              />
              <div>
                <span className="block text-white font-medium">{post.author}</span>
                <span className="block text-white/80">{post.date}</span>
              </div>
            </div>
            <div className="flex items-center text-white/80">
              <Clock className="h-5 w-5 mr-2" />
              <span>{post.readTime} min read</span>
            </div>
          </div>
          <a 
            href={`/article/${post.id}`}
            className="group inline-flex items-center px-6 py-3 text-lg font-medium text-indigo-700 dark:text-indigo-900 bg-white rounded-xl hover:bg-gray-50 transition-colors"
          >
            Read Article
            <ArrowRight className="ml-2 h-5 w-5 transform transition-transform group-hover:translate-x-1" />
          </a>
        </div>
        <div className="lg:w-1/2">
          <div className="relative h-full min-h-[400px] lg:min-h-0">
            <img 
              src={post.imageUrl} 
              alt={post.title}
              className="absolute inset-0 w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-r from-indigo-600/50 to-transparent lg:hidden" />
          </div>
        </div>
      </div>
    </article>
  );
}