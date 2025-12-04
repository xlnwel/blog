import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    // 如果仓库名是 username.github.io，base 应该是 '/'
    // 否则 base 应该是 '/仓库名/'
    // 可以通过环境变量 VITE_BASE_PATH 来设置，默认为 '/'
    // 在 GitHub Actions 中，如果仓库名不是 username.github.io，需要设置 VITE_BASE_PATH=/仓库名/
    const base = process.env.VITE_BASE_PATH || '/';
    
    return {
      base: base,
      server: {
        port: 3000,
        host: '0.0.0.0',
      },
      plugins: [react()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
