import React from 'react';
import { Download, MapPin, Mail, Linkedin, Github, ExternalLink, Sparkles } from 'lucide-react';

const Resume: React.FC = () => {
  const glassCardClass = "bg-white/60 backdrop-blur-lg border border-white/40 shadow-lg shadow-slate-200/50 rounded-3xl p-6 transition-all hover:bg-white/70";

  return (
    <div className="animate-in fade-in duration-700 space-y-6">
      
      {/* Top Grid: Hero & Contact */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Profile Card */}
        <div className={`${glassCardClass} lg:col-span-2 flex flex-col justify-between relative overflow-hidden group`}>
          <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
            <Sparkles size={120} />
          </div>
          <div>
            <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-2 tracking-tight">Jane Doe</h1>
            <h2 className="text-xl text-primary-600 font-medium mb-6">Senior Full Stack Engineer</h2>
            <p className="text-slate-600 leading-relaxed max-w-xl">
               Bridging the gap between complex technical systems and strategic business goals. 
               Scaling React applications and optimizing revenue streams through data-driven development.
            </p>
          </div>
          <div className="mt-8 flex gap-3">
             <button className="flex items-center gap-2 px-5 py-2.5 bg-slate-900 text-white rounded-full hover:bg-slate-800 transition-all hover:scale-105 shadow-lg shadow-slate-900/20 active:scale-95 text-sm font-medium">
                <Download size={16} />
                Resume PDF
             </button>
             <button className="flex items-center gap-2 px-5 py-2.5 bg-white text-slate-700 border border-slate-200 rounded-full hover:bg-slate-50 transition-all hover:scale-105 active:scale-95 text-sm font-medium shadow-sm">
                Contact Me
             </button>
          </div>
        </div>

        {/* Contact/Socials "Widget" */}
        <div className={`${glassCardClass} flex flex-col justify-center gap-4`}>
          <div className="flex items-center gap-3 text-slate-600 p-2 rounded-xl hover:bg-white/50 transition-colors">
            <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center text-blue-600">
               <MapPin size={20} />
            </div>
            <div>
              <p className="text-xs text-slate-400 uppercase font-semibold">Location</p>
              <p className="font-medium">San Francisco, CA</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3 text-slate-600 p-2 rounded-xl hover:bg-white/50 transition-colors">
             <div className="w-10 h-10 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600">
               <Mail size={20} />
            </div>
             <div>
              <p className="text-xs text-slate-400 uppercase font-semibold">Email</p>
              <p className="font-medium">jane.doe@example.com</p>
            </div>
          </div>

          <div className="flex gap-3 mt-2 pl-2">
             <a href="#" className="w-10 h-10 flex items-center justify-center rounded-full bg-slate-100 hover:bg-slate-200 hover:scale-110 transition-all text-slate-700">
                <Linkedin size={20} />
             </a>
             <a href="#" className="w-10 h-10 flex items-center justify-center rounded-full bg-slate-100 hover:bg-slate-200 hover:scale-110 transition-all text-slate-700">
                <Github size={20} />
             </a>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Experience Column */}
        <div className="lg:col-span-2 space-y-6">
          <section className={glassCardClass}>
            <h3 className="text-lg font-bold text-slate-900 mb-6 flex items-center gap-2">
              Experience
            </h3>
            
            <div className="space-y-8">
              {/* Job 1 */}
              <div className="relative pl-8 before:absolute before:left-0 before:top-2 before:bottom-0 before:w-[2px] before:bg-gradient-to-b before:from-primary-400 before:to-slate-200">
                <div className="absolute left-[-5px] top-2 w-3 h-3 rounded-full bg-primary-500 ring-4 ring-white"></div>
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className="text-lg font-bold text-slate-900">Senior Frontend Engineer</h4>
                    <p className="text-primary-600 font-medium">TechCorp Inc.</p>
                  </div>
                  <span className="text-xs font-semibold text-slate-500 bg-slate-100/80 px-3 py-1 rounded-full border border-slate-200/50">2021 - Present</span>
                </div>
                <ul className="space-y-2 text-slate-600 text-sm mt-3">
                  <li className="flex items-start gap-2">
                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-slate-400 flex-shrink-0"></span>
                    Architected a micro-frontend architecture using module federation.
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-slate-400 flex-shrink-0"></span>
                    Led team of 6 engineers to rebuild customer dashboard.
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-slate-400 flex-shrink-0"></span>
                    Implemented rigorous E2E testing with Cypress.
                  </li>
                </ul>
              </div>

              {/* Job 2 */}
              <div className="relative pl-8 before:absolute before:left-0 before:top-2 before:bottom-0 before:w-[2px] before:bg-slate-200">
                <div className="absolute left-[-5px] top-2 w-3 h-3 rounded-full bg-slate-400 ring-4 ring-white"></div>
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className="text-lg font-bold text-slate-900">Product Developer</h4>
                    <p className="text-primary-600 font-medium">StartUp Studio</p>
                  </div>
                  <span className="text-xs font-semibold text-slate-500 bg-slate-100/80 px-3 py-1 rounded-full border border-slate-200/50">2018 - 2021</span>
                </div>
                <ul className="space-y-2 text-slate-600 text-sm mt-3">
                  <li className="flex items-start gap-2">
                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-slate-400 flex-shrink-0"></span>
                    Developed MVP for a fintech app using React Native.
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-slate-400 flex-shrink-0"></span>
                    Collaborated with founders on revenue strategies.
                  </li>
                </ul>
              </div>
            </div>
          </section>

          <section className={glassCardClass}>
             <h3 className="text-lg font-bold text-slate-900 mb-4">Projects</h3>
             <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white/50 hover:bg-white/80 transition-colors p-4 rounded-2xl border border-white/40 shadow-sm cursor-pointer group">
                    <div className="flex justify-between items-center mb-2">
                        <h4 className="font-bold text-slate-900 group-hover:text-primary-600 transition-colors">OpenSource UI</h4>
                        <ExternalLink size={16} className="text-slate-400" />
                    </div>
                    <p className="text-xs text-slate-600 mb-3 leading-relaxed">A lightweight, accessible component library built with Headless UI.</p>
                    <div className="flex gap-2">
                        <span className="text-[10px] uppercase font-bold text-slate-500 bg-slate-100 px-2 py-1 rounded-md">React</span>
                        <span className="text-[10px] uppercase font-bold text-slate-500 bg-slate-100 px-2 py-1 rounded-md">TS</span>
                    </div>
                </div>
                <div className="bg-white/50 hover:bg-white/80 transition-colors p-4 rounded-2xl border border-white/40 shadow-sm cursor-pointer group">
                     <div className="flex justify-between items-center mb-2">
                        <h4 className="font-bold text-slate-900 group-hover:text-primary-600 transition-colors">FinDash</h4>
                        <ExternalLink size={16} className="text-slate-400" />
                    </div>
                    <p className="text-xs text-slate-600 mb-3 leading-relaxed">Real-time financial dashboard visualizing crypto assets.</p>
                    <div className="flex gap-2">
                        <span className="text-[10px] uppercase font-bold text-slate-500 bg-slate-100 px-2 py-1 rounded-md">D3.js</span>
                        <span className="text-[10px] uppercase font-bold text-slate-500 bg-slate-100 px-2 py-1 rounded-md">Next</span>
                    </div>
                </div>
             </div>
          </section>
        </div>

        {/* Sidebar Column */}
        <div className="space-y-6">
          <section className={`${glassCardClass} h-fit`}>
            <h3 className="text-sm font-bold text-slate-900 mb-4 uppercase tracking-wider opacity-70">Tech Stack</h3>
            <div className="space-y-5">
              <div>
                <span className="text-xs font-semibold text-slate-500 mb-2 block">Frontend</span>
                <div className="flex flex-wrap gap-2">
                    {['React', 'TypeScript', 'Tailwind', 'Next.js', 'Redux'].map(skill => (
                        <span key={skill} className="px-3 py-1.5 bg-indigo-50/50 border border-indigo-100/50 text-indigo-700 rounded-lg text-xs font-semibold">{skill}</span>
                    ))}
                </div>
              </div>
              <div>
                <span className="text-xs font-semibold text-slate-500 mb-2 block">Backend</span>
                <div className="flex flex-wrap gap-2">
                    {['Node.js', 'Python', 'PostgreSQL', 'GraphQL'].map(skill => (
                        <span key={skill} className="px-3 py-1.5 bg-emerald-50/50 border border-emerald-100/50 text-emerald-700 rounded-lg text-xs font-semibold">{skill}</span>
                    ))}
                </div>
              </div>
            </div>
          </section>

          <section className={`${glassCardClass} h-fit`}>
            <h3 className="text-sm font-bold text-slate-900 mb-4 uppercase tracking-wider opacity-70">Education</h3>
            <div className="space-y-4">
                <div className="group">
                    <h4 className="text-sm font-bold text-slate-900">M.S. Comp Sci</h4>
                    <p className="text-xs text-slate-500">Stanford University</p>
                    <p className="text-xs text-slate-400 mt-1 bg-slate-100/50 inline-block px-2 py-0.5 rounded">2016 - 2018</p>
                </div>
                <div className="w-full h-px bg-slate-100"></div>
                <div className="group">
                    <h4 className="text-sm font-bold text-slate-900">B.S. Economics</h4>
                    <p className="text-xs text-slate-500">UC Berkeley</p>
                    <p className="text-xs text-slate-400 mt-1 bg-slate-100/50 inline-block px-2 py-0.5 rounded">2012 - 2016</p>
                </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default Resume;