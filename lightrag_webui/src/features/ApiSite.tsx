import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { useTabVisibility } from '@/contexts/useTabVisibility'
import { backendBaseUrl } from '@/lib/constants'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'

export default function ApiSite() {
  const { t } = useTranslation()
  const { isTabVisible } = useTabVisibility()
  const isApiTabVisible = isTabVisible('api')
  const [iframeLoaded, setIframeLoaded] = useState(false)

  useEffect(() => {
    if (!iframeLoaded) {
      setIframeLoaded(true)
    }
  }, [iframeLoaded])

  return (
    <div className={`size-full min-h-0 ${isApiTabVisible ? '' : 'hidden'}`}>
      <div className="flex h-full min-h-0 flex-col gap-4 overflow-hidden p-4">
        <Card className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <CardHeader className="shrink-0">
            <CardTitle>{t('apiSite.swaggerTitle', 'Swagger')}</CardTitle>
            <CardDescription>
              {t(
                'apiSite.swaggerDescription',
                'OpenAPI documentation is loaded from the running backend at `/docs`. If a newly added endpoint is missing here, restart the backend process first.'
              )}
            </CardDescription>
          </CardHeader>
          <CardContent className="min-h-0 flex-1 overflow-hidden p-0">
            {iframeLoaded ? (
              <iframe
                src={backendBaseUrl + '/docs'}
                className="size-full"
                style={{ border: 'none' }}
                key="api-docs-iframe"
              />
            ) : (
              <div className="flex h-full w-full items-center justify-center bg-background">
                <div className="text-center">
                  <div className="mb-2 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                  <p>{t('apiSite.loading')}</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
