import http from 'node:http'
import path from 'node:path'
import { existsSync } from 'node:fs'
import { promises as fs } from 'node:fs'

export type Language = 'en' | 'zh'

export type ScreenshotTarget = {
  language: Language
  routePath: string
  outputName: string
}

export const repoRoot = process.cwd()
export const docsRoot = path.resolve(repoRoot, 'docs')
export const distRoot = path.resolve(repoRoot, 'docs/.vitepress/dist')
export const readmeScreenshotsRoot = path.resolve(docsRoot, 'public/screenshots/readme')
export const docsBasePath = normalizeBasePath(
  process.env.DOCS_BASE_PATH || '/learn-world-model/'
)

const MIME_TYPES: Record<string, string> = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2'
}

export const readmeScreenshotTargets: ScreenshotTarget[] = [
  { language: 'en', routePath: '/en/', outputName: 'en-home.png' },
  {
    language: 'en',
    routePath: '/en/lectures/lecture-01-internal-simulation/',
    outputName: 'en-lecture-01.png'
  },
  {
    language: 'en',
    routePath: '/en/lectures/lecture-03-architecture-patterns/01-architectures',
    outputName: 'en-lecture-03.png'
  }
]

export async function ensureDirectory(targetPath: string) {
  await fs.mkdir(targetPath, { recursive: true })
}

export function toAbsoluteSiteUrl(origin: string, routePath: string) {
  const normalizedOrigin = origin.replace(/\/$/, '')
  const normalizedRoute = routePath.startsWith('/') ? routePath : `/${routePath}`
  return `${normalizedOrigin}${docsBasePath.replace(/\/$/, '')}${normalizedRoute}`
}

export async function startStaticServer(rootDir = distRoot) {
  const server = http.createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url || '/', 'http://127.0.0.1')
      const filePath = resolveStaticFilePath(rootDir, requestUrl.pathname)

      if (!filePath) {
        response.statusCode = 404
        response.end('Not Found')
        return
      }

      const data = await fs.readFile(filePath)
      const ext = path.extname(filePath)
      response.setHeader('Content-Type', MIME_TYPES[ext] || 'application/octet-stream')
      response.end(data)
    } catch (error) {
      response.statusCode = 500
      response.end(error instanceof Error ? error.message : 'Unknown server error')
    }
  })

  const address = await new Promise<http.AddressInfo>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const currentAddress = server.address()
      if (!currentAddress || typeof currentAddress === 'string') {
        reject(new Error('Failed to bind static export server'))
        return
      }
      resolve(currentAddress)
    })
  })

  return {
    origin: `http://127.0.0.1:${address.port}`,
    close: async () =>
      await new Promise<void>((resolve, reject) => {
        server.close((error) => {
          if (error) {
            reject(error)
            return
          }
          resolve()
        })
      })
  }
}

function normalizeBasePath(input: string) {
  const withLeadingSlash = input.startsWith('/') ? input : `/${input}`
  return withLeadingSlash.endsWith('/') ? withLeadingSlash : `${withLeadingSlash}/`
}

function resolveStaticFilePath(rootDir: string, pathname: string) {
  const normalizedPath = pathname.replace(/\/+/g, '/')
  if (!normalizedPath.startsWith(docsBasePath)) {
    return null
  }

  const relativePath = normalizedPath.slice(docsBasePath.length).replace(/^\/+/, '')
  const candidates = buildPathCandidates(rootDir, relativePath)
  return candidates.find((candidate) => existsSync(candidate)) || null
}

function buildPathCandidates(rootDir: string, relativePath: string) {
  if (!relativePath) {
    return [path.join(rootDir, 'index.html')]
  }

  if (relativePath.endsWith('/')) {
    return [path.join(rootDir, relativePath, 'index.html')]
  }

  if (path.extname(relativePath)) {
    return [path.join(rootDir, relativePath)]
  }

  return [
    path.join(rootDir, `${relativePath}.html`),
    path.join(rootDir, relativePath, 'index.html')
  ]
}
