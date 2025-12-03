import React, { useState, useMemo } from 'react';
import { Post } from '../types';
import { Search, Tag, ArrowRight, BookOpen, Filter } from 'lucide-react';

interface BlogFeedProps {
  category: 'technical' | 'business';
  posts: Post[];
  onSelectPost: (post: Post) => void;
}

const BlogFeed: React.FC<BlogFeedProps> = ({ category, posts, onSelectPost }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTag, setSelectedTag] = useState<string | null>(null);

  const allTags = useMemo(() => {
    const tags = new Set<string>();
    posts.forEach(post => post.tags.forEach(tag => tags.add(tag)));
    return Array.from(tags);
  }, [posts]);

  const filteredPosts = useMemo(() => {
    return posts.filter(post => {
      const matchesSearch = post.title.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesTag = selectedTag ? post.tags.includes(selectedTag) : true;
      return matchesSearch && matchesTag;
    });
  }, [posts, searchQuery, selectedTag]);

  const themeColor = category === 'technical' ? 'indigo' : 'emerald';
  // Use explicit classes for color variants to avoid template literal purge issues
  const accentText = category === 'technical' ? 'text-indigo-600' : 'text-emerald-600';
  const accentBg = category === 'technical' ? 'bg-indigo-500' : 'bg-emerald-500';

  return (
    <div className="animate-in fade-in duration-700 grid grid-cols-1 lg:grid-cols-4 gap-8 items-start">
      
      {/* Sidebar Filters - macOS Sidebar Style */}
      <div className="lg:col-span-1">
        <div className="bg-white/50 backdrop-blur-xl p-5 rounded-3xl border border-white/40 shadow-lg shadow-slate-200/50 sticky top-32">
            <div className="mb-6">
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 px-2">Search</h3>
                <div className="relative group">
                    <input
                        type="text"
                        placeholder="Search..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-9 pr-4 py-2 bg-white/60 border border-slate-200/60 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-200 focus:bg-white transition-all text-sm shadow-sm"
                    />
                    <Search className="absolute left-3 top-2.5 text-slate-400 group-focus-within:text-slate-600 transition-colors" size={16} />
                </div>
            </div>

            <div>
                <div className="flex items-center justify-between mb-2 px-2">
                     <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Tags</h4>
                     {selectedTag && (
                        <button onClick={() => setSelectedTag(null)} className="text-[10px] text-slate-500 hover:text-red-500 font-medium">Clear</button>
                     )}
                </div>
               
                <div className="space-y-1">
                    <button
                        onClick={() => setSelectedTag(null)}
                        className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                            selectedTag === null 
                            ? 'bg-slate-200/50 text-slate-900' 
                            : 'text-slate-600 hover:bg-white/40'
                        }`}
                    >
                        All Posts
                    </button>
                    {allTags.map(tag => (
                        <button
                            key={tag}
                            onClick={() => setSelectedTag(selectedTag === tag ? null : tag)}
                            className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 capitalize ${
                                selectedTag === tag
                                ? `${accentBg} text-white shadow-md shadow-${themeColor}-500/20` 
                                : 'text-slate-600 hover:bg-white/40'
                            }`}
                        >
                            <Tag size={12} className={selectedTag === tag ? 'opacity-100' : 'opacity-40'} />
                            {tag}
                        </button>
                    ))}
                </div>
            </div>
        </div>
      </div>

      {/* Main Feed */}
      <div className="lg:col-span-3 space-y-6">
        <div className="px-2">
            <h2 className="text-3xl font-bold text-slate-900 capitalize tracking-tight">
                {category}
            </h2>
            <p className="text-slate-500 font-medium">
                {category === 'technical' 
                    ? 'Engineering deep dives & tutorials.' 
                    : 'Strategy, growth, and product notes.'}
            </p>
        </div>

        {filteredPosts.length === 0 ? (
            <div className="bg-white/40 backdrop-blur-md p-12 rounded-3xl border border-white/40 text-center shadow-sm">
                <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center mx-auto mb-4 text-slate-300">
                    <Filter size={32} />
                </div>
                <h3 className="text-lg font-bold text-slate-900">No results found</h3>
                <p className="text-slate-500 mt-1 mb-6">We couldn't find any posts matching your criteria.</p>
                <button 
                    onClick={() => { setSearchQuery(''); setSelectedTag(null); }}
                    className="px-6 py-2 bg-slate-900 text-white rounded-full text-sm font-medium hover:scale-105 transition-transform"
                >
                    Clear Filters
                </button>
            </div>
        ) : (
            <div className="grid gap-6">
                {filteredPosts.map(post => (
                    <div 
                        key={post.id} 
                        onClick={() => onSelectPost(post)}
                        className="bg-white/60 backdrop-blur-lg p-6 md:p-8 rounded-3xl border border-white/40 shadow-lg shadow-slate-200/50 hover:shadow-xl hover:scale-[1.01] hover:bg-white/80 transition-all duration-300 cursor-pointer group relative overflow-hidden"
                    >
                         {/* Decorative gradient blob on hover */}
                        <div className={`absolute -right-20 -top-20 w-64 h-64 bg-gradient-to-br from-${themeColor}-100 to-transparent rounded-full blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none`}></div>

                        <div className="relative z-10">
                            <div className="flex items-center gap-3 mb-4">
                                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">{post.date}</span>
                                <div className="h-1 w-1 bg-slate-300 rounded-full"></div>
                                <div className="flex gap-2">
                                    {post.tags.slice(0, 2).map(tag => (
                                        <span key={tag} className={`text-[10px] font-bold px-2 py-1 rounded-md uppercase tracking-wide bg-white border border-slate-100 ${accentText}`}>
                                            {tag}
                                        </span>
                                    ))}
                                </div>
                            </div>
                            
                            <h3 className="text-2xl font-bold text-slate-900 group-hover:text-slate-700 transition-colors mb-3 tracking-tight">
                                {post.title}
                            </h3>
                            
                            <p className="text-slate-600 mb-6 line-clamp-2 leading-relaxed">
                                {post.excerpt}
                            </p>
                            
                            <div className={`flex items-center text-sm font-bold ${accentText} group-hover:translate-x-2 transition-transform`}>
                                Read Article <ArrowRight size={16} className="ml-2" />
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        )}
      </div>
    </div>
  );
};

export default BlogFeed;