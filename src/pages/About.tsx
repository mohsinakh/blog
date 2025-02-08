import React from "react";
import {
  Code2,
  Brain,
  Database,
  Coffee,
  Heart,
  Music,
  Book,
  Gamepad2,
  Camera,
  Plane,
} from "lucide-react";
import mohsinImg from "../../mohsin.png";

const About: React.FC = () => {
  const skills = [
    { name: "Frontend", icon: Code2, level: 95 },
    { name: "Backend", icon: Database, level: 90 },
    { name: "AI/ML", icon: Brain, level: 85 },
  ];

  const interests = [
    { icon: Coffee, label: "Coffee Enthusiast" },
    { icon: Music, label: "Music Listening" },
    { icon: Book, label: "Reading" },
    { icon: Gamepad2, label: "Gaming" },
    { icon: Camera, label: "Photography" },
    { icon: Plane, label: "Traveling" },
  ];

  return (
    <div className="min-h-screen pt-20 pb-12 bg-gradient-to-br from-gray-100 to-white dark:from-gray-900 dark:to-black font-sans">
      {/* Hero Section */}
      <div className="container mx-auto pt-20 px-6 sm:px-8 md:px-12">
        <div className="relative">
          <div className="absolute inset-0 bg-blue-600 dark:bg-blue-500 transform -skew-y-6 z-0"></div>
          <div className="relative z-10 py-20 text-center text-white">
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold mb-6 tracking-tight">
              Hey, I'm <span className="text-blue-200">Mohsin Abbas</span>
            </h1>
            <p className="text-lg sm:text-xl max-w-3xl mx-auto font-light leading-relaxed tracking-wide">
              A tech geek, problem solver, and full-stack developer who loves to
              build cool things on the internet.
            </p>
          </div>
        </div>

        {/* Main Content */}
        <div className="mt-24 sm:mt-20 grid grid-cols-1 md:grid-cols-2 gap-12">
          {/* Left Column */}
          <div className="space-y-8">
            {/* Journey Section */}
            <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6 sm:p-8 shadow-lg transform hover:scale-[1.02] transition-all">
              <h2 className="text-2xl sm:text-3xl font-bold mb-6 tracking-tight text-gray-900 dark:text-white">
                My{" "}
                <span className="text-blue-600 dark:text-blue-400">
                  Journey
                </span>
              </h2>
              <div className="prose dark:prose-invert max-w-none space-y-6">
                <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
                  <span className="text-xl font-semibold text-blue-600 dark:text-blue-400">
                    "Hello, World!"
                  </span>
                  That’s where it all began. What started as simple curiosity
                  quickly turned into an obsession with
                  <strong>building, automating, and creating</strong>. From
                  writing tiny scripts that made my life easier to developing
                  full-fledged applications, every line of code I wrote fueled
                  my excitement for technology.
                </p>

                <div className="border-l-4 border-blue-600 dark:border-blue-400 pl-4">
                  <p className="italic text-gray-600 dark:text-gray-400">
                    <em>
                      Frontend, backend, AI—if it’s about coding, I want to
                      explore it.
                    </em>
                  </p>
                </div>

                <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
                  But blogging? That was never part of the plan. Initially, it
                  was just a personal <strong>tech journal</strong>, a way to
                  document what I was learning. Somewhere along the way, I
                  realized that if my notes helped me, they might help others
                  too.
                </p>

                <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
                  Now, writing is more of a <strong>free-time hobby</strong>—a
                  space where I share my thoughts on tech, coding challenges,
                  and random ideas that spark my interest. When I’m not deep in
                  code (or writing about it), you’ll probably find me:
                </p>

                <ul className="list-disc pl-6 text-gray-700 dark:text-gray-300">
                  <li>Experimenting with new frameworks & tools</li>
                  <li>Building side projects just for fun</li>
                  <li>Reading up on the latest in tech</li>
                  <li>
                    Or just unwinding with a <strong>good playlist</strong>
                  </li>
                </ul>
              </div>
            </div>

            {/* Skills Section */}
            <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6 sm:p-8 shadow-lg">
              <h2 className="text-2xl sm:text-3xl font-bold mb-6 tracking-tight text-gray-900 dark:text-white">
                Technical{" "}
                <span className="text-blue-600 dark:text-blue-400">Skills</span>
              </h2>
              <div className="space-y-6">
                {skills.map((skill, index) => {
                  const Icon = skill.icon;
                  return (
                    <div key={index}>
                      <div className="flex items-center gap-3 mb-2">
                        <Icon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                        <span className="font-semibold tracking-wide text-gray-800 dark:text-white">
                          {skill.name}
                        </span>
                        <span className="ml-auto font-mono text-blue-600 dark:text-blue-400">
                          {skill.level}%
                        </span>
                      </div>
                      <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-600 dark:bg-blue-400 rounded-full transition-all duration-1000"
                          style={{ width: `${skill.level}%` }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            {/* Personal Info */}
            <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-8 shadow-lg">
              <div className="relative">
                <img
                  src={mohsinImg}
                  alt="Mohsin Abbas"
                  className="w-[200px] h-[200px] object-cover rounded-full mx-auto border-4 border-white dark:border-gray-800 shadow-lg"
                />
                <div className="absolute bottom-0 right-1/2 transform translate-x-20 translate-y-4">
                  <Heart className="w-8 h-8 text-red-500 fill-current animate-pulse" />
                </div>
              </div>
              <div className="text-center mt-6">
                <h2 className="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-blue-600 to-blue-400 dark:from-blue-400 dark:to-blue-300 text-transparent bg-clip-text">
                  Mohsin Abbas
                </h2>
                <p className="text-xl text-gray-600 dark:text-gray-400 mt-2 font-light tracking-wide">
                  Full Stack Developer | AI Enthusiast
                </p>
                <p className="text-blue-600 dark:text-blue-400 mt-1 font-medium tracking-wide">
                  India
                </p>

                
                {/* Portfolio & LinkedIn Links */}
                  <div className="flex justify-center mt-4 space-x-4">
                    <a
                      href="https://mohsinabbas.site"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 dark:text-blue-400 font-semibold hover:underline"
                    >
                      🌐 Portfolio
                    </a>
                    <a
                      href="https://www.linkedin.com/in/mohsin-abbas-7252b126b/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 dark:text-blue-400 font-semibold hover:underline"
                    >
                      🔗 LinkedIn
                    </a>
                  </div>
              </div>
            </div>
              </div>
            </div>

            {/* Interests */}
            <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6 sm:p-8 shadow-lg">
              <h2 className="text-2xl sm:text-3xl font-bold mb-6 tracking-tight text-gray-900 dark:text-white">
                Beyond{" "}
                <span className="text-blue-600 dark:text-blue-400">Coding</span>
              </h2>
              <div className="grid grid-cols-2 gap-4 sm:gap-6">
                {interests.map((interest, index) => {
                  const Icon = interest.icon;
                  return (
                    <div
                      key={index}
                      className="flex items-center gap-2 sm:gap-3 p-3 sm:p-4 bg-gray-100 dark:bg-gray-700/50 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors"
                    >
                      <Icon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                      <span className="text-xs sm:text-base font-medium tracking-wide text-gray-800 dark:text-white">
                        {interest.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Quote */}
            <div className="bg-blue-600 dark:bg-blue-500 rounded-xl p-8 shadow-lg text-white">
              <blockquote className="text-2xl italic font-light tracking-wide leading-relaxed">
                "The best way to predict the future is to create it."
              </blockquote>
              <p className="mt-4 font-medium tracking-wider text-blue-200">
                - My coding philosophy
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
