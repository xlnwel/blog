import React from 'react';
import { Briefcase, Code, User, Layers } from 'lucide-react';
import { ViewState } from '../types';

interface HeaderProps {
  currentView: ViewState;
  onChangeView: (view: ViewState) => void;
}

const Header: React.FC<HeaderProps> = ({ currentView, onChangeView }) => {
  const navItems: { id: ViewState; label: string; icon: React.ReactNode }[] = [
    { id: 'resume', label: 'Resume', icon: <User size={18} /> },
    { id: 'tech', label: 'Tech', icon: <Code size={18} /> },
    { id: 'product', label: 'Products', icon: <Layers size={18} /> },
    { id: 'business', label: 'Business', icon: <Briefcase size={18} /> },
  ];

  return (
    <div className="fixed top-6 left-0 right-0 z-50 flex justify-center px-4">
      <nav className="bg-white/70 backdrop-blur-xl border border-white/40 shadow-xl shadow-slate-200/50 rounded-full px-2 py-2 flex items-center gap-1 sm:gap-2 ring-1 ring-white/50">
        
        <div 
          className="flex-shrink-0 cursor-pointer flex items-center gap-2 pr-4 pl-2 border-r border-slate-200/50 mr-1"
          onClick={() => onChangeView('resume')}
        >
          <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-700 rounded-full flex items-center justify-center text-white font-bold shadow-md shadow-primary-500/30 text-xs">
            JD
          </div>
          <span className="font-semibold text-sm tracking-tight text-slate-800 hidden sm:block">
            Jane Doe
          </span>
        </div>
        
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onChangeView(item.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 ease-out ${
              currentView === item.id
                ? 'bg-white shadow-sm text-slate-900 ring-1 ring-black/5'
                : 'text-slate-500 hover:text-slate-900 hover:bg-white/50'
            }`}
          >
            <span className={currentView === item.id ? 'text-primary-600' : ''}>
              {item.icon}
            </span>
            <span className={currentView === item.id ? 'block' : 'hidden sm:block'}>
              {item.label}
            </span>
          </button>
        ))}
      </nav>
    </div>
  );
};

export default Header;