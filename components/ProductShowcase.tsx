import React from 'react';
import { ExternalLink, Github, Zap, BarChart, Cloud, Smartphone, ArrowRight, ShieldCheck } from 'lucide-react';

interface Product {
  id: string;
  name: string;
  tagline: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  tags: string[];
  link?: string;
  github?: string;
}

const ProductShowcase: React.FC = () => {
  const products: Product[] = [
    {
      id: '1',
      name: 'FinDash',
      tagline: 'Crypto Intelligence',
      description: 'Real-time financial dashboard visualizing crypto assets with predictive analytics.',
      icon: <BarChart size={32} className="text-white" />,
      color: 'bg-blue-500',
      tags: ['D3.js', 'React', 'Finance'],
      link: '#',
      github: '#'
    },
    {
      id: '2',
      name: 'CloudSync',
      tagline: 'Seamless Storage',
      description: 'A distributed file storage system utilizing IPFS for redundant, secure data management.',
      icon: <Cloud size={32} className="text-white" />,
      color: 'bg-indigo-500',
      tags: ['IPFS', 'Go', 'Distributed'],
      link: '#',
      github: '#'
    },
    {
      id: '3',
      name: 'NativeUI',
      tagline: 'Component Library',
      description: 'An accessible, lightweight component library built for React Native and Expo.',
      icon: <Smartphone size={32} className="text-white" />,
      color: 'bg-purple-500',
      tags: ['React Native', 'UI/UX'],
      link: '#',
      github: '#'
    },
    {
      id: '4',
      name: 'SecureNet',
      tagline: 'VPN Wrapper',
      description: 'Zero-config VPN wrapper ensuring secure connections for remote teams.',
      icon: <ShieldCheck size={32} className="text-white" />,
      color: 'bg-emerald-500',
      tags: ['Security', 'Rust', 'Networking'],
      link: '#',
      github: '#'
    },
    {
      id: '5',
      name: 'BoltStream',
      tagline: 'Low Latency Video',
      description: 'WebRTC based streaming platform optimized for high-frequency trading data.',
      icon: <Zap size={32} className="text-white" />,
      color: 'bg-amber-500',
      tags: ['WebRTC', 'Socket.io'],
      link: '#',
      github: '#'
    }
  ];

  return (
    <div className="animate-in fade-in duration-700 space-y-8">
      <div className="px-2 mb-8">
        <h2 className="text-3xl font-bold text-slate-900 tracking-tight">Products</h2>
        <p className="text-slate-500 font-medium">Tools and applications I've shipped.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {products.map((product) => (
          <div 
            key={product.id}
            className="bg-white/60 backdrop-blur-lg p-6 rounded-3xl border border-white/40 shadow-lg shadow-slate-200/50 hover:shadow-xl hover:scale-[1.02] transition-all duration-300 group flex flex-col h-full"
          >
            <div className="flex items-start justify-between mb-6">
              <div className={`w-16 h-16 rounded-2xl ${product.color} shadow-lg shadow-${product.color}/30 flex items-center justify-center shrink-0`}>
                {product.icon}
              </div>
              <div className="flex gap-2">
                 {product.github && (
                   <a href={product.github} className="p-2 bg-white/50 hover:bg-white rounded-full text-slate-500 hover:text-slate-900 transition-colors">
                     <Github size={18} />
                   </a>
                 )}
              </div>
            </div>

            <div className="mb-4 flex-grow">
              <h3 className="text-xl font-bold text-slate-900 mb-1">{product.name}</h3>
              <p className="text-xs font-bold text-slate-400 uppercase tracking-wide mb-3">{product.tagline}</p>
              <p className="text-slate-600 text-sm leading-relaxed">{product.description}</p>
            </div>

            <div className="mt-auto">
               <div className="flex flex-wrap gap-2 mb-4">
                  {product.tags.map(tag => (
                      <span key={tag} className="px-2 py-1 bg-white/50 border border-white/60 rounded-md text-[10px] font-bold text-slate-500 uppercase tracking-wide">
                        {tag}
                      </span>
                  ))}
               </div>
               
               <a 
                 href={product.link}
                 className="flex items-center justify-between w-full px-4 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-sm font-bold transition-colors group-hover:bg-primary-50 group-hover:text-primary-700"
               >
                 <span>View Details</span>
                 <ArrowRight size={16} />
               </a>
            </div>
          </div>
        ))}

        {/* 'Coming Soon' Placeholder Card */}
        <div className="bg-white/30 backdrop-blur-sm p-6 rounded-3xl border border-white/20 border-dashed flex flex-col items-center justify-center text-center min-h-[300px] group cursor-default">
            <div className="w-16 h-16 rounded-2xl bg-slate-100 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <span className="text-3xl text-slate-300">+</span>
            </div>
            <h3 className="text-lg font-bold text-slate-400">New Project</h3>
            <p className="text-sm text-slate-400 mt-2">Currently in development</p>
        </div>
      </div>
    </div>
  );
};

export default ProductShowcase;