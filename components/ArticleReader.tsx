import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { ChevronLeft, Calendar, Clock } from 'lucide-react';
import { Post } from '../types';

interface ArticleReaderProps {
  post: Post;
  onBack: () => void;
}

const ArticleReader: React.FC<ArticleReaderProps> = ({ post, onBack }) => {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchContent = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(post.file);
        if (!response.ok) {
          throw new Error('Failed to load article content');
        }
        const text = await response.text();
        setContent(text);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchContent();
  }, [post]);

  return (
    <div className="animate-in slide-in-from-bottom-8 duration-700 max-w-3xl mx-auto relative">
      <button 
        onClick={onBack}
        className="absolute -left-16 top-2 hidden xl:flex flex-col items-center justify-center w-10 h-10 bg-white/40 hover:bg-white backdrop-blur-md rounded-full text-slate-500 hover:text-slate-800 transition-all hover:scale-110 shadow-sm border border-white/20"
        title="Go Back"
      >
        <ChevronLeft size={20} />
      </button>

      <button 
        onClick={onBack}
        className="xl:hidden mb-4 flex items-center text-slate-500 hover:text-primary-600 transition-colors font-medium bg-white/40 px-3 py-1 rounded-full w-fit backdrop-blur-sm"
      >
        <ChevronLeft size={18} className="mr-1" /> Back
      </button>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-32 bg-white/60 backdrop-blur-xl rounded-3xl border border-white/40 shadow-xl">
          <div className="w-8 h-8 border-4 border-slate-200 border-t-slate-800 rounded-full animate-spin mb-4"></div>
          <p className="text-slate-500 font-medium">Loading...</p>
        </div>
      ) : error ? (
        <div className="bg-red-50/80 backdrop-blur-md text-red-600 p-8 rounded-3xl text-center border border-red-100 shadow-lg">
            <h3 className="font-bold mb-2">Error Loading Post</h3>
            <p>{error}</p>
        </div>
      ) : (
        <article className="bg-white/80 backdrop-blur-2xl rounded-[2rem] shadow-2xl shadow-slate-300/50 border border-white/60 overflow-hidden ring-1 ring-black/5">
          <div className="p-8 md:p-12 border-b border-slate-200/50 bg-gradient-to-b from-white to-white/50">
            <div className="flex gap-2 mb-6">
                {post.tags.map(tag => (
                    <span key={tag} className="px-3 py-1 bg-slate-100/50 border border-slate-200 text-xs font-bold text-slate-600 rounded-lg uppercase tracking-wider shadow-sm">
                        {tag}
                    </span>
                ))}
            </div>
            <h1 className="text-3xl md:text-5xl font-bold text-slate-900 mb-6 leading-tight tracking-tight">{post.title}</h1>
            <div className="flex items-center gap-6 text-slate-500 text-sm font-medium">
                <span className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-50 border border-slate-100"><Calendar size={14} /> {post.date}</span>
                {post.readTime && <span className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-50 border border-slate-100"><Clock size={14} /> {post.readTime}</span>}
            </div>
          </div>
          
          <div className="p-8 md:p-12">
            <div className="markdown-body">
              <ReactMarkdown>{content}</ReactMarkdown>
            </div>
          </div>
        </article>
      )}
    </div>
  );
};

export default ArticleReader;