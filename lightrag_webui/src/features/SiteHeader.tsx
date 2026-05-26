import { useCallback, useEffect, useMemo, useState } from 'react'
import Button from '@/components/ui/Button'
import { SiteInfo, webuiPrefix } from '@/lib/constants'
import AppSettings from '@/components/AppSettings'
import { TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { useSettingsStore } from '@/stores/settings'
import { useAuthStore, useBackendState } from '@/stores/state'
import { cn } from '@/lib/utils'
import { useTranslation } from 'react-i18next'
import { navigationService } from '@/services/navigation'
import { FileTextIcon, ZapIcon, GithubIcon, LogOutIcon } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/Dialog'
import Textarea from '@/components/ui/Textarea'
import { getWorkspaces, updateWorkspaceDescription } from '@/api/lightrag'

interface NavigationTabProps {
  value: string
  currentTab: string
  children: React.ReactNode
}

function NavigationTab({ value, currentTab, children }: NavigationTabProps) {
  return (
    <TabsTrigger
      value={value}
      className={cn(
        'cursor-pointer px-2 py-1 transition-all',
        currentTab === value ? '!bg-emerald-400 !text-zinc-50' : 'hover:bg-background/60'
      )}
    >
      {children}
    </TabsTrigger>
  )
}

function TabsNavigation() {
  const currentTab = useSettingsStore.use.currentTab()
  const { t } = useTranslation()

  return (
    <div className="flex h-8 self-center">
      <TabsList className="h-full gap-2">
        <NavigationTab value="documents" currentTab={currentTab}>
          {t('header.documents')}
        </NavigationTab>
        <NavigationTab value="chunks" currentTab={currentTab}>
          {t('header.chunks', 'Chunks')}
        </NavigationTab>
        <NavigationTab value="knowledge-graph" currentTab={currentTab}>
          {t('header.knowledgeGraph')}
        </NavigationTab>
        <NavigationTab value="retrieval" currentTab={currentTab}>
          {t('header.retrieval')}
        </NavigationTab>
        <NavigationTab value="api" currentTab={currentTab}>
          {t('header.api')}
        </NavigationTab>
      </TabsList>
    </div>
  )
}

function WorkspaceSelector() {
  const { t } = useTranslation()
  const workspace = useSettingsStore.use.workspace()
  const defaultWorkspace = useSettingsStore.use.defaultWorkspace()
  const availableWorkspaces = useSettingsStore.use.availableWorkspaces()
  const setWorkspace = useSettingsStore.use.setWorkspace()
  const setWorkspaceInfo = useSettingsStore.use.setWorkspaceInfo()
  const [isLoading, setIsLoading] = useState(false)
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [descriptionDraft, setDescriptionDraft] = useState('')
  const [isSaving, setIsSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)

  const workspaceOptions = useMemo(() => {
    if (availableWorkspaces.length > 0) {
      return availableWorkspaces
    }
    return [{ id: defaultWorkspace, alias: defaultWorkspace }]
  }, [availableWorkspaces, defaultWorkspace])

  const selectedWorkspace =
    workspace && workspaceOptions.some((item) => item.id === workspace)
      ? workspace
      : defaultWorkspace
  const selectedWorkspaceOption = workspaceOptions.find((item) => item.id === selectedWorkspace)
  const selectedWorkspaceDescription = selectedWorkspaceOption?.description || ''

  useEffect(() => {
    let cancelled = false

    const loadWorkspaces = async () => {
      setIsLoading(true)
      try {
        const response = await getWorkspaces()
        if (!cancelled) {
          setWorkspaceInfo(response.workspaces || [], response.default_workspace || 'default')
        }
      } catch (error) {
        console.warn('Failed to load workspaces:', error)
      } finally {
        if (!cancelled) {
          setIsLoading(false)
        }
      }
    }

    loadWorkspaces()
    return () => {
      cancelled = true
    }
  }, [setWorkspaceInfo])

  const handleWorkspaceChange = useCallback((value: string) => {
    if (value === defaultWorkspace) {
      setWorkspace(null)
    } else {
      setWorkspace(value)
    }
    useBackendState.getState().resetHealthCheckTimer()
  }, [defaultWorkspace, setWorkspace])

  const handleOpenDescriptionDialog = useCallback(() => {
    setDescriptionDraft(selectedWorkspaceDescription)
    setSaveError(null)
    setIsDialogOpen(true)
  }, [selectedWorkspaceDescription])

  const handleSaveDescription = useCallback(async () => {
    setIsSaving(true)
    setSaveError(null)
    try {
      const updated = await updateWorkspaceDescription(selectedWorkspace, descriptionDraft)
      const nextWorkspaces = workspaceOptions.map((item) =>
        item.id === updated.id
          ? {
            ...item,
            alias: updated.alias,
            description: updated.description,
            has_description: updated.has_description
          }
          : item
      )
      setWorkspaceInfo(nextWorkspaces, defaultWorkspace)
      setIsDialogOpen(false)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setSaveError(message)
      console.warn('Failed to save workspace description:', error)
    } finally {
      setIsSaving(false)
    }
  }, [defaultWorkspace, descriptionDraft, selectedWorkspace, setWorkspaceInfo, workspaceOptions])

  return (
    <div className="flex items-center gap-1">
      <div className="w-[180px]">
        <span className="sr-only">{t('header.workspace', 'Workspace')}</span>
        <Select
          value={selectedWorkspace}
          onValueChange={handleWorkspaceChange}
          disabled={isLoading || workspaceOptions.length === 0}
        >
          <SelectTrigger className="h-8 text-xs">
            <SelectValue placeholder={t('header.workspace', 'Workspace')} />
          </SelectTrigger>
          <SelectContent>
            {workspaceOptions.map((item) => (
              <SelectItem key={item.id} value={item.id}>
                {item.id === defaultWorkspace
                  ? `${item.alias} (${t('header.defaultWorkspace', 'default')})`
                  : item.alias}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <Button
        variant="ghost"
        size="icon"
        className="size-8"
        side="bottom"
        tooltip={selectedWorkspaceDescription || t('header.workspaceDescription', 'Workspace description')}
        onClick={handleOpenDescriptionDialog}
        disabled={isLoading || workspaceOptions.length === 0}
      >
        <FileTextIcon className="size-4" aria-hidden="true" />
      </Button>
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>{t('header.workspaceDescription', 'Workspace description')}</DialogTitle>
            <DialogDescription>
              {selectedWorkspaceOption?.alias || selectedWorkspace}
            </DialogDescription>
          </DialogHeader>
          <Textarea
            value={descriptionDraft}
            onChange={(event) => setDescriptionDraft(event.target.value)}
            placeholder={t('header.workspaceDescriptionPlaceholder', 'Describe what this workspace contains.')}
            className="min-h-32"
            disabled={isSaving}
          />
          {saveError && (
            <p className="text-destructive text-sm">{saveError}</p>
          )}
          <DialogFooter>
            <Button variant="ghost" onClick={() => setIsDialogOpen(false)} disabled={isSaving}>
              {t('common.cancel', 'Cancel')}
            </Button>
            <Button onClick={handleSaveDescription} disabled={isSaving}>
              {isSaving ? t('common.saving', 'Saving...') : t('common.save', 'Save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default function SiteHeader() {
  const { t } = useTranslation()
  const { isGuestMode, coreVersion, apiVersion, username, webuiTitle, webuiDescription } = useAuthStore()

  const versionDisplay = (coreVersion && apiVersion)
    ? `${coreVersion}/${apiVersion}`
    : null;

  // Check if frontend needs rebuild (apiVersion ends with warning symbol)
  const hasWarning = apiVersion?.endsWith('⚠️');
  const versionTooltip = hasWarning
    ? t('header.frontendNeedsRebuild')
    : versionDisplay ? `v${versionDisplay}` : '';

  const handleLogout = () => {
    navigationService.navigateToLogin();
  }

  return (
    <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
      <div className="min-w-[200px] w-auto flex items-center">
        <a href={webuiPrefix} className="flex items-center gap-2">
          <ZapIcon className="size-4 text-emerald-400" aria-hidden="true" />
          <span className="font-bold md:inline-block">{SiteInfo.name}</span>
        </a>
        {webuiTitle && (
          <div className="flex items-center">
            <span className="mx-1 text-xs text-gray-500 dark:text-gray-400">|</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="font-medium text-sm cursor-default">
                    {webuiTitle}
                  </span>
                </TooltipTrigger>
                {webuiDescription && (
                  <TooltipContent side="bottom">
                    {webuiDescription}
                  </TooltipContent>
                )}
              </Tooltip>
            </TooltipProvider>
          </div>
        )}
      </div>

      <div className="flex h-10 flex-1 items-center justify-center">
        <TabsNavigation />
        {isGuestMode && (
          <div className="ml-2 self-center px-2 py-1 text-xs bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200 rounded-md">
            {t('login.guestMode', 'Guest Mode')}
          </div>
        )}
      </div>

      <nav className="min-w-[400px] w-auto flex items-center justify-end">
        <div className="flex items-center gap-2">
          <WorkspaceSelector />
          {versionDisplay && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-xs text-gray-500 dark:text-gray-400 mr-1 cursor-default">
                    v{versionDisplay}
                  </span>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  {versionTooltip}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          <Button variant="ghost" size="icon" side="bottom" tooltip={t('header.projectRepository')}>
            <a href={SiteInfo.github} target="_blank" rel="noopener noreferrer">
              <GithubIcon className="size-4" aria-hidden="true" />
            </a>
          </Button>
          <AppSettings />
          {!isGuestMode && (
            <Button
              variant="ghost"
              size="icon"
              side="bottom"
              tooltip={`${t('header.logout')} (${username})`}
              onClick={handleLogout}
            >
              <LogOutIcon className="size-4" aria-hidden="true" />
            </Button>
          )}
        </div>
      </nav>
    </header>
  )
}
