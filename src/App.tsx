import { Routes, Route } from "react-router"; // Removed `useLocation`
import Header from "./components/Header";
import Home from "./pages/Home";
import Articles from "./pages/Articles";
import Article from "./pages/Article";
import About from "./pages/About";
import { ThemeProvider } from "./context/ThemeContext";

const App = () => {
  return (

    <ThemeProvider>

    <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
      {/* Header */}
      <Header />

      {/* Main Content */}
      <main className="flex-grow container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/articles" element={<Articles />} />
          <Route path="/about" element={<About />} />
          <Route path="/article/:id" element={<Article />} />
          
        </Routes>
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-center text-gray-600 dark:text-gray-400 text-sm">
          <p>© {new Date().getFullYear()} DevBlog. All rights reserved.</p>
        </div>
      </footer>
    </div>

    </ThemeProvider>
  );
};



export default App;
