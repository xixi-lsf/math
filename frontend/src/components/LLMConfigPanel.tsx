import { useState } from 'react'
import type { LLMConfig } from '../config/llmConfig'
import { PRESET_PROVIDERS, saveConfig } from '../config/llmConfig'
import { Settings, Check, X, ChevronDown } from 'lucide-react'
import clsx from 'clsx'

interface Props {
  config: LLMConfig
  onChange: (config: LLMConfig) => void
}

export default function LLMConfigPanel({ config, onChange }: Props) {
  const [open, setOpen] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; msg: string } | null>(null)
  const [local, setLocal] = useState<LLMConfig>(config)

  const handleSave = () => {
    saveConfig(local)
    onChange(local)
    setOpen(false)
    setTestResult(null)
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const res = await fetch('/api/v1/config/validate-llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(local),
      })
      const data = await res.json()
      setTestResult({ ok: data.valid, msg: data.valid ? `连接成功：${data.reply}` : data.error })
    } catch (e) {
      setTestResult({ ok: false, msg: String(e) })
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-900 transition-colors"
      >
        <Settings size={15} />
        <span className="font-medium">{config.model}</span>
        <ChevronDown size={13} className={clsx('transition-transform', open && 'rotate-180')} />
      </button>

      {open && (
        <div className="absolute right-0 top-8 z-50 w-80 bg-white rounded-xl border border-slate-200 shadow-lg p-4 space-y-3">
          <h3 className="font-semibold text-sm text-slate-800">LLM 配置</h3>

          {/* Presets */}
          <div>
            <label className="text-xs text-slate-500 mb-1 block">快速选择</label>
            <div className="flex flex-wrap gap-1.5">
              {PRESET_PROVIDERS.map(p => (
                <button
                  key={p.name}
                  onClick={() => setLocal(prev => ({ ...prev, base_url: p.base_url, model: p.model }))}
                  className="text-xs px-2 py-1 rounded border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50 transition-colors"
                >
                  {p.name}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <Field label="Base URL" value={local.base_url} onChange={v => setLocal(p => ({ ...p, base_url: v }))} />
            <Field label="API Key" value={local.api_key} onChange={v => setLocal(p => ({ ...p, api_key: v }))} type="password" />
            <Field label="Model" value={local.model} onChange={v => setLocal(p => ({ ...p, model: v }))} />
          </div>

          {testResult && (
            <div className={clsx(
              'text-xs px-3 py-2 rounded-lg flex items-start gap-1.5',
              testResult.ok ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
            )}>
              {testResult.ok ? <Check size={12} className="mt-0.5 flex-shrink-0" /> : <X size={12} className="mt-0.5 flex-shrink-0" />}
              {testResult.msg}
            </div>
          )}

          <div className="flex gap-2 pt-1">
            <button
              onClick={handleTest}
              disabled={testing}
              className="flex-1 text-xs py-2 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors disabled:opacity-50"
            >
              {testing ? '测试中...' : '测试连接'}
            </button>
            <button
              onClick={handleSave}
              className="flex-1 text-xs py-2 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition-colors"
            >
              保存
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function Field({ label, value, onChange, type = 'text' }: {
  label: string; value: string; onChange: (v: string) => void; type?: string
}) {
  return (
    <div>
      <label className="text-xs text-slate-500 block mb-0.5">{label}</label>
      <input
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full text-xs px-2.5 py-1.5 rounded-lg border border-slate-200 focus:outline-none focus:border-indigo-400 bg-slate-50"
      />
    </div>
  )
}
