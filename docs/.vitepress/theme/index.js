import DefaultTheme from 'vitepress/theme'
import './style.css'

let rootEnglishLabel = 'English'
let mermaidViewer
let mermaidViewerBody
let mermaidViewerStage
let mermaidObserver
let mermaidViewerUrl
let mermaidViewerImage
let mermaidViewerNaturalWidth = 0
let mermaidViewerNaturalHeight = 0
let mermaidViewerFitScale = 1
let mermaidViewerScale = 1

function dedupeLocaleMenuLinks() {
  if (typeof document === 'undefined') return

  const containers = document.querySelectorAll(
    '.VPNavBarTranslations .items, .VPNavBarExtra .group.translations, .VPNavScreenTranslations .list'
  )

  for (const container of containers) {
    const seen = new Set()
    const links = container.querySelectorAll('a[href]')

    for (const link of links) {
      const label = link.textContent?.trim() || ''
      const href = link.getAttribute('href') || ''
      const key = `${label}::${href}`

      if (seen.has(key)) {
        const removable = link.closest('.VPMenuLink, .item')
        removable?.remove()
        continue
      }

      seen.add(key)
    }
  }
}

function queueLocaleMenuDedupe() {
  if (typeof window === 'undefined') return
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      dedupeLocaleMenuLinks()
    })
  })
}

function syncRootLocaleLabel(siteData, href = '') {
  const locales = siteData.value?.locales
  if (!locales?.root || !locales?.zh) return

  if (locales.root.label && locales.root.label !== locales.zh.label) {
    rootEnglishLabel = locales.root.label
  }

  if (locales.en?.label) {
    rootEnglishLabel = locales.en.label
  }

  const base = siteData.value.base || '/'
  const normalized = href || (typeof window !== 'undefined' ? window.location.pathname : '')
  const relative = normalized.startsWith(base)
    ? normalized.slice(base.length - 1)
    : normalized

  locales.root.label = relative.startsWith('/zh/') ? locales.zh.label : rootEnglishLabel
}

function ensureMermaidViewer() {
  if (typeof document === 'undefined') return
  if (mermaidViewer && mermaidViewerBody && mermaidViewerStage) return

  mermaidViewer = document.createElement('div')
  mermaidViewer.className = 'mermaid-viewer'
  mermaidViewer.innerHTML = `
    <div class="mermaid-viewer__backdrop" data-mermaid-close="true"></div>
    <div class="mermaid-viewer__panel" role="dialog" aria-modal="true" aria-label="Expanded Mermaid diagram">
      <div class="mermaid-viewer__toolbar">
        <button type="button" class="mermaid-viewer__tool" aria-label="Zoom out" data-mermaid-action="zoom-out">−</button>
        <button type="button" class="mermaid-viewer__tool mermaid-viewer__tool--label" aria-label="Reset zoom" data-mermaid-action="reset">100%</button>
        <button type="button" class="mermaid-viewer__tool" aria-label="Zoom in" data-mermaid-action="zoom-in">+</button>
        <button type="button" class="mermaid-viewer__close" aria-label="Close diagram" data-mermaid-close="true">×</button>
      </div>
      <div class="mermaid-viewer__body">
        <div class="mermaid-viewer__stage"></div>
      </div>
    </div>
  `

  mermaidViewerBody = mermaidViewer.querySelector('.mermaid-viewer__body')
  mermaidViewerStage = mermaidViewer.querySelector('.mermaid-viewer__stage')

  mermaidViewer.addEventListener('click', (event) => {
    const target = event.target
    if (!(target instanceof HTMLElement)) return

    if (
      target.dataset.mermaidClose === 'true' ||
      target === mermaidViewerBody ||
      target === mermaidViewerStage
    ) {
      closeMermaidViewer()
      return
    }

    const action = target.dataset.mermaidAction
    if (action === 'zoom-in') {
      setMermaidViewerScale(mermaidViewerScale + 0.2)
    } else if (action === 'zoom-out') {
      setMermaidViewerScale(mermaidViewerScale - 0.2)
    } else if (action === 'reset') {
      setMermaidViewerScale(1)
    }
  })

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && mermaidViewer?.classList.contains('is-open')) {
      closeMermaidViewer()
    } else if ((event.key === '+' || event.key === '=') && mermaidViewer?.classList.contains('is-open')) {
      setMermaidViewerScale(mermaidViewerScale + 0.2)
    } else if (event.key === '-' && mermaidViewer?.classList.contains('is-open')) {
      setMermaidViewerScale(mermaidViewerScale - 0.2)
    }
  })

  mermaidViewerBody?.addEventListener('wheel', (event) => {
    if (!mermaidViewer?.classList.contains('is-open')) return
    event.preventDefault()
    const delta = event.deltaY < 0 ? 0.12 : -0.12
    setMermaidViewerScale(mermaidViewerScale + delta)
  }, { passive: false })

  window.addEventListener('resize', () => {
    if (mermaidViewer?.classList.contains('is-open')) {
      updateMermaidViewerFitScale()
      setMermaidViewerScale(mermaidViewerScale)
    }
  })

  document.body.appendChild(mermaidViewer)
}

function serializeMermaidSvg(svg) {
  const clone = svg.cloneNode(true)
  if (!(clone instanceof SVGElement)) return null

  if (!clone.getAttribute('xmlns')) {
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  }
  if (!clone.getAttribute('xmlns:xlink')) {
    clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink')
  }

  const rect = svg.getBoundingClientRect()
  const viewBox = svg.viewBox?.baseVal
  const rawWidth = rect.width || viewBox?.width || Number.parseFloat(svg.getAttribute('width') || '') || 1200
  const rawHeight = rect.height || viewBox?.height || Number.parseFloat(svg.getAttribute('height') || '') || 800

  if (!clone.getAttribute('viewBox')) {
    clone.setAttribute('viewBox', `0 0 ${Math.max(1, Math.round(rawWidth))} ${Math.max(1, Math.round(rawHeight))}`)
  }

  clone.setAttribute('width', String(Math.max(1, Math.round(rawWidth))))
  clone.setAttribute('height', String(Math.max(1, Math.round(rawHeight))))

  return {
    markup: new XMLSerializer().serializeToString(clone),
    width: Math.max(1, rawWidth),
    height: Math.max(1, rawHeight)
  }
}

function updateMermaidViewerFitScale() {
  if (!mermaidViewerNaturalWidth || !mermaidViewerNaturalHeight || typeof window === 'undefined') return

  const maxWidth = Math.max(320, window.innerWidth - 16)
  const maxHeight = Math.max(240, window.innerHeight - 16)
  mermaidViewerFitScale = Math.max(
    0.25,
    Math.min(maxWidth / mermaidViewerNaturalWidth, maxHeight / mermaidViewerNaturalHeight)
  )
}

function setMermaidViewerScale(nextScale) {
  if (!mermaidViewerStage || !mermaidViewerNaturalWidth || !mermaidViewerNaturalHeight) return

  mermaidViewerScale = Math.max(0.5, Math.min(nextScale, 4))

  const width = mermaidViewerNaturalWidth * mermaidViewerFitScale * mermaidViewerScale
  const height = mermaidViewerNaturalHeight * mermaidViewerFitScale * mermaidViewerScale

  mermaidViewerStage.style.width = `${width}px`
  mermaidViewerStage.style.height = `${height}px`

  const resetButton = mermaidViewer?.querySelector('[data-mermaid-action="reset"]')
  if (resetButton instanceof HTMLElement) {
    resetButton.textContent = `${Math.round(mermaidViewerScale * 100)}%`
  }
}

function openMermaidViewer(diagram) {
  ensureMermaidViewer()
  if (!mermaidViewerBody || !mermaidViewer || !mermaidViewerStage) return

  const svg = diagram.querySelector('svg')
  if (!svg) return

  if (mermaidViewerUrl) {
    URL.revokeObjectURL(mermaidViewerUrl)
    mermaidViewerUrl = undefined
  }

  const serialized = serializeMermaidSvg(svg)
  if (!serialized) return

  const blob = new Blob([serialized.markup], { type: 'image/svg+xml;charset=utf-8' })
  mermaidViewerUrl = URL.createObjectURL(blob)

  const image = document.createElement('img')
  image.className = 'mermaid-viewer__image'
  image.alt = 'Expanded Mermaid diagram'
  image.src = mermaidViewerUrl
  image.addEventListener('load', () => {
    mermaidViewerNaturalWidth = serialized.width
    mermaidViewerNaturalHeight = serialized.height
    updateMermaidViewerFitScale()
    setMermaidViewerScale(1)
    if (mermaidViewerBody) {
      mermaidViewerBody.scrollTo({ top: 0, left: 0 })
    }
  })

  mermaidViewerImage = image
  mermaidViewerStage.replaceChildren(image)
  mermaidViewer.classList.add('is-open')
  document.documentElement.classList.add('has-mermaid-viewer')
}

function closeMermaidViewer() {
  if (!mermaidViewer || !mermaidViewerBody || !mermaidViewerStage) return
  mermaidViewer.classList.remove('is-open')
  mermaidViewerStage.replaceChildren()
  mermaidViewerStage.style.width = ''
  mermaidViewerStage.style.height = ''
  if (mermaidViewerUrl) {
    URL.revokeObjectURL(mermaidViewerUrl)
    mermaidViewerUrl = undefined
  }
  mermaidViewerImage = undefined
  mermaidViewerNaturalWidth = 0
  mermaidViewerNaturalHeight = 0
  mermaidViewerFitScale = 1
  mermaidViewerScale = 1
  document.documentElement.classList.remove('has-mermaid-viewer')
}

function bindMermaidZoomables(root = document) {
  if (typeof document === 'undefined') return
  ensureMermaidViewer()

  const diagrams = root.querySelectorAll('.vp-doc .mermaid')
  for (const diagram of diagrams) {
    if (!(diagram instanceof HTMLElement) || diagram.dataset.mermaidZoomBound === 'true') {
      continue
    }

    diagram.dataset.mermaidZoomBound = 'true'
    diagram.tabIndex = 0
    diagram.setAttribute('role', 'button')
    diagram.setAttribute('aria-label', 'Open Mermaid diagram in fullscreen')

    diagram.addEventListener('click', () => {
      openMermaidViewer(diagram)
    })

    diagram.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        openMermaidViewer(diagram)
      }
    })
  }
}

function queueMermaidBinding() {
  if (typeof window === 'undefined') return
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      bindMermaidZoomables()
    })
  })
}

function ensureMermaidObserver() {
  if (typeof document === 'undefined' || mermaidObserver) return
  mermaidObserver = new MutationObserver(() => {
    bindMermaidZoomables()
  })

  mermaidObserver.observe(document.body, {
    childList: true,
    subtree: true
  })
}

export default {
  extends: DefaultTheme,
  enhanceApp(ctx) {
    syncRootLocaleLabel(ctx.siteData)
    queueLocaleMenuDedupe()
    queueMermaidBinding()
    ensureMermaidObserver()

    const previous = ctx.router.onAfterRouteChange
    ctx.router.onAfterRouteChange = async (to) => {
      syncRootLocaleLabel(ctx.siteData, to)
      queueLocaleMenuDedupe()
      closeMermaidViewer()
      queueMermaidBinding()
      if (previous) {
        await previous(to)
      }
    }
  }
}
