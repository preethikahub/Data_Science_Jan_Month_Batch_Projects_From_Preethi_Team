import { useState } from 'react';
import { FileCode2, Copy, Check, Terminal, FileText, Download } from 'lucide-react';
import { pythonSourceRegistry } from '../utils';

export default function CodeHub() {
  const [activeFile, setActiveFile] = useState<keyof typeof pythonSourceRegistry>('preprocessing.py');
  const [copied, setCopied] = useState<boolean>(false);

  const handleCopy = () => {
    const code = pythonSourceRegistry[activeFile];
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const fileMeta = {
    'preprocessing.py': { desc: 'EDA & Feature preprocessors (scaling, encoding, SMOTE oversampling)', type: 'python' },
    'model.py': { desc: 'Benchmarking Suite for nine classification algorithms & serialization', type: 'python' },
    'train.py': { desc: 'Main pipeline coordinator executing fits, scores, and saving diagnostic plots', type: 'python' },
    'predict.py': { desc: 'Inference engine executing single predictions & local risk factor analysis', type: 'python' },
    'app.py': { desc: 'Flask backend server hosting REST API endpoints and Bootstrap pages', type: 'python' },
    'requirements.txt': { desc: 'Version-locked dependency lists for pip installer installations', type: 'text' },
    'README.md': { desc: 'Project configuration, architecture overview, installation, and deployment guides', type: 'markdown' },
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6" id="code-hub-root">
      <div>
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <Terminal className="w-5 h-5 text-indigo-600" />
          Production Python Code Workspace
        </h2>
        <p className="text-xs text-slate-500 mt-1">
          Access the complete, PEP 8 compliant ML Python stack. Browse, copy, or review modular code blocks.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left Hand Directory Selector */}
        <div className="lg:col-span-4 space-y-2 border-r border-slate-200 pr-0 lg:pr-6">
          <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-4 px-2">
            Modules Explorer
          </h3>
          {(Object.keys(pythonSourceRegistry) as Array<keyof typeof pythonSourceRegistry>).map((fileName) => {
            const isSelected = activeFile === fileName;
            const isText = fileName.endsWith('.txt') || fileName.endsWith('.md');

            return (
              <button
                key={fileName}
                onClick={() => {
                  setActiveFile(fileName);
                  setCopied(false);
                }}
                className={`w-full text-left p-3 rounded-xl text-xs font-bold flex items-center justify-between transition-all ${
                  isSelected
                    ? 'bg-indigo-50 text-indigo-700 border-l-4 border-l-indigo-600'
                    : 'text-slate-600 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center space-x-2.5 truncate">
                  {isText ? (
                    <FileText className="w-4 h-4 shrink-0 opacity-85" />
                  ) : (
                    <FileCode2 className="w-4 h-4 shrink-0 text-indigo-500 opacity-85" />
                  )}
                  <span className="truncate">{fileName}</span>
                </div>
              </button>
            );
          })}

          <div className="mt-8 bg-slate-50 border border-slate-200 rounded-xl p-4 text-[11px] text-slate-500 leading-relaxed">
            <p className="font-bold text-slate-700 mb-1 flex items-center gap-1">
              <Download className="w-3.5 h-3.5" /> Local Execution Note:
            </p>
            Copy these scripts into your local project environment. Execute <code>python train.py</code> to trigger model fit and evaluate the local predictions.
          </div>
        </div>

        {/* Right Hand Code Viewer Panel */}
        <div className="lg:col-span-8 flex flex-col min-h-[460px] border border-slate-800 rounded-xl overflow-hidden bg-slate-900 shadow-inner">
          {/* Header controls */}
          <div className="bg-slate-950/85 px-5 py-3 flex items-center justify-between border-b border-slate-800">
            <div>
              <span className="text-xs font-bold text-slate-200">{activeFile}</span>
              <p className="text-[10px] text-slate-400 mt-0.5 truncate max-w-sm md:max-w-lg">
                {fileMeta[activeFile].desc}
              </p>
            </div>
            <button
              onClick={handleCopy}
              className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy Code</span>
                </>
              )}
            </button>
          </div>

          {/* Preformatted Content */}
          <div className="flex-1 overflow-auto p-5 font-mono text-[11px] leading-relaxed text-slate-300">
            <pre className="whitespace-pre">
              <code>{pythonSourceRegistry[activeFile]}</code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
