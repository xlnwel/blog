import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Resume from './components/Resume';
import BlogFeed from './components/BlogFeed';
import ArticleReader from './components/ArticleReader';
import ProductShowcase from './components/ProductShowcase';
import { ViewState, Post } from './types';

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<ViewState>('resume');
  const [selectedPost, setSelectedPost] = useState<Post | null>(null);
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch posts manifest
    fetch('posts.json')
      .then((res) => res.json())
      .then((data) => {
        setPosts(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to fetch posts:', err);
        setLoading(false);
      });
  }, []);

  const handleViewChange = (view: ViewState) => {
    setCurrentView(view);
    setSelectedPost(null); // Reset selection when changing main tabs
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handlePostSelect = (post: Post) => {
    setSelectedPost(post);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleBackToFeed = () => {
    setSelectedPost(null);
  };

  // Filter posts based on current view
  const techPosts = posts.filter((p) => p.category === 'technical');
  const businessPosts = posts.filter((p) => p.category === 'business');

  return (
    <div className="flex flex-col font-sans min-h-screen">
      <Header currentView={currentView} onChangeView={handleViewChange} />

      {/* Added more top padding to account for floating header */}
      <main className="flex-grow max-w-6xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-24 md:py-32">
        
        {selectedPost ? (
          <ArticleReader post={selectedPost} onBack={handleBackToFeed} />
        ) : (
          <>
            {currentView === 'resume' && <Resume />}
            {currentView === 'product' && <ProductShowcase />}
            
            {loading && currentView !== 'resume' && currentView !== 'product' && (
                <div className="flex flex-col items-center justify-center py-20">
                     <div className="w-8 h-8 border-4 border-slate-200 border-t-slate-800 rounded-full animate-spin mb-4"></div>
                     <p className="text-slate-500 font-medium">Loading Content...</p>
                </div>
            )}

            {!loading && currentView === 'tech' && (
              <BlogFeed 
                category="technical" 
                posts={techPosts} 
                onSelectPost={handlePostSelect} 
              />
            )}

            {!loading && currentView === 'business' && (
              <BlogFeed 
                category="business" 
                posts={businessPosts} 
                onSelectPost={handlePostSelect} 
              />
            )}
          </>
        )}

      </main>

      <footer className="border-t border-slate-200/50 mt-auto py-8 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-4 text-center text-slate-400 text-sm font-medium">
          <p>&copy; {new Date().getFullYear()} Jane Doe. Designed with React & Tailwind.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;