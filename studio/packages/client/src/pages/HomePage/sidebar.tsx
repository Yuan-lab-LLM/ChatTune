import { memo, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { matchPath, useLocation, useNavigate } from "react-router-dom";
import {
  BeakerIcon,
  DatabaseIcon,
  LayoutDashboardIcon,
  ListIcon,
  LineChartIcon,
  CpuIcon,
  SettingsIcon,
  LogOutIcon,
  UsersIcon,
  KeyRoundIcon,
  CircleUserRoundIcon,
  ChevronUpIcon,
} from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarSeparator,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar.tsx";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu.tsx";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip.tsx";
import SettingsDialog from "./Settings";
import { getSidebarItems } from "./config";
import { RunPageSection, useStudioSidebar } from "@/context/SidebarContext.tsx";
import { normalizeWandbUrl, useWandb } from "@/context/WandbContext.tsx";
import { WandbMonitorDialog } from "@/components/wandb/WandbMonitorDialog.tsx";
import { useAuth } from "@/context/AuthContext.tsx";
import UserManagementDialog from "./UserManagementDialog";
import ChangePasswordDialog from "./ChangePasswordDialog";

const StudioSidebar = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { t } = useTranslation();
  const { user, isAdmin, logout } = useAuth();

  const currentUsername = user?.username || "";

  const { wandbLinks, hasWandbLinks, getUserWandbInfo, hasUserWandbUrl } =
    useWandb();

  // Get current user's wandb info
  const userWandbInfo = getUserWandbInfo(currentUsername);
  const userHasWandbUrl = hasUserWandbUrl(currentUsername);

  // Debug logging
  useEffect(() => {
  }, [currentUsername, userWandbInfo, userHasWandbUrl, hasWandbLinks]);

  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [wandbDialogOpen, setWandbDialogOpen] = useState(false);
  const [userManagementDialogOpen, setUserManagementDialogOpen] =
    useState(false);
  const [changePasswordDialogOpen, setChangePasswordDialogOpen] =
    useState(false);

  const sidebarItems = getSidebarItems(t);
  const {
    showRunPageNavigation,
    isRunPagePanelOpen,
    runPageSection,
    setRunPagePanelOpen,
    setRunPageSection,
  } = useStudioSidebar();

  const runPageMatch =
    matchPath("/projects/:projectName/*", location.pathname) ||
    matchPath("/projects/:projectName", location.pathname);
  const projectName = runPageMatch?.params?.projectName;
  const isLandingRoute =
    location.pathname === "/" ||
    location.pathname === "/home" ||
    location.pathname === "/projects";

  const runPageNavItems: Array<{
    key: RunPageSection;
    icon: typeof ListIcon;
    label: string;
  }> = [
    {
      key: "overview",
      icon: LayoutDashboardIcon,
      label: t("overview.tab.overview"),
    },
    ...(isAdmin
      ? [{ key: "runs" as RunPageSection, icon: ListIcon, label: t("overview.tab.runs") }]
      : []),
    {
      key: "datasets",
      icon: DatabaseIcon,
      label: t("tab.datasets") || "数据管理",
    },
    { key: "models", icon: CpuIcon, label: t("tab.models") || "模型管理" },
    {
      key: "evaluation",
      icon: BeakerIcon,
      label: t("tab.evaluation") || "评测管理",
    },
  ] as Array<{
    key: RunPageSection;
    icon: typeof ListIcon;
    label: string;
  }>;

  const handleRunPageNavClick = (section: RunPageSection) => {
    if (runPageSection === section) {
      setRunPagePanelOpen(!isRunPagePanelOpen);
    } else {
      setRunPageSection(section);
      setRunPagePanelOpen(true);
    }
    if (projectName) {
      const projectRunsPath = `/projects/${projectName}/runs`;
      if (
        location.pathname !== projectRunsPath &&
        !location.pathname.startsWith(`${projectRunsPath}/`)
      ) {
        navigate(projectRunsPath);
      }
    }
  };

  return (
    <Sidebar collapsible="icon" className="studio-sidebar-shell">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              size="lg"
              asChild
              tooltip="MedFlow ChatTune"
              className="studio-sidebar-brand"
            >
              <a
                href="/"
                onClick={(event) => {
                  event.preventDefault();
                  navigate("/");
                }}
              >
                <div className="studio-sidebar-brand-mark flex aspect-square size-10 items-center justify-center rounded-2xl bg-primary text-primary-foreground font-bold text-[1.45rem]">
                  M
                </div>
                <div className="grid flex-1 text-left leading-tight">
                  <span className="truncate studio-sidebar-brand-title">
                    MedFlow
                  </span>
                  <span className="truncate studio-sidebar-brand-subtitle">
                    ChatTune
                  </span>
                </div>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        {showRunPageNavigation && (
          <SidebarGroup
            className="-mt-2 runpage-global-nav"
            data-runpage-nav="true"
          >
            <SidebarGroupContent>
              <SidebarMenu>
                {runPageNavItems.map((item) => (
                  <SidebarMenuItem key={item.key}>
                    <SidebarMenuButton
                      className="cursor-pointer studio-sidebar-nav-button"
                      tooltip={item.label}
                      isActive={
                        runPageSection === item.key && isRunPagePanelOpen
                      }
                      data-section={item.key}
                      onClick={() => handleRunPageNavClick(item.key)}
                    >
                      <item.icon />
                      <span>{item.label}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}
        {sidebarItems.map((item) => (
          <SidebarGroup key={item.title} className="-mt-2">
            <SidebarGroupLabel>
              <span>{item.title}</span>
            </SidebarGroupLabel>
            {item.items.map((subItem) => (
              <SidebarGroupContent key={subItem.title}>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      className="cursor-pointer studio-sidebar-nav-button"
                      tooltip={subItem.title}
                      onClick={() => {
                        // Check if it's an external URL
                        if (
                          subItem.url?.startsWith("http://") ||
                          subItem.url?.startsWith("https://")
                        ) {
                          window.open(
                            subItem.url,
                            "_blank",
                            "noopener,noreferrer",
                          );
                        } else {
                          // Handle internal routes
                          navigate(subItem.url);
                        }
                      }}
                    >
                      <subItem.icon />
                      <span>{subItem.title}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            ))}
          </SidebarGroup>
        ))}
      </SidebarContent>
      {/* Footer with settings and wandb monitor */}
      <SidebarFooter>
        {showRunPageNavigation && <SidebarSeparator className="mx-0 my-1" />}
        <SidebarMenu className="studio-sidebar-footer-menu">
          <SidebarMenuItem
            className={`studio-sidebar-footer-quick-actions ${
              isLandingRoute ? "studio-sidebar-footer-quick-actions-single" : ""
            }`}
          >
            {!isLandingRoute && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="studio-sidebar-footer-tooltip-trigger">
                    <SidebarMenuButton
                      className={`studio-sidebar-nav-button studio-sidebar-footer-button ${
                        userWandbInfo.pending ||
                        (!userHasWandbUrl && !hasWandbLinks)
                          ? "opacity-50"
                          : ""
                      }`}
                      onClick={() => {
                        if (userHasWandbUrl && userWandbInfo.url) {
                          window.open(
                            normalizeWandbUrl(userWandbInfo.url),
                            "_blank",
                            "noopener,noreferrer",
                          );
                        } else if (hasWandbLinks) {
                          setWandbDialogOpen(true);
                        }
                      }}
                      disabled={
                        userWandbInfo.pending ||
                        (!userHasWandbUrl && !hasWandbLinks)
                      }
                    >
                      <LineChartIcon />
                      <span>Wandb</span>
                    </SidebarMenuButton>
                  </span>
                </TooltipTrigger>
                <TooltipContent side="top">
                  {userWandbInfo.pending
                    ? t("auth.wandbLoading")
                    : userHasWandbUrl
                      ? t("auth.wandbOpen")
                      : t("auth.wandbUnavailable")}
                </TooltipContent>
              </Tooltip>
            )}
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="studio-sidebar-footer-tooltip-trigger">
                  <SidebarMenuButton
                    isActive={settingsDialogOpen}
                    className="studio-sidebar-nav-button studio-sidebar-footer-button"
                    onClick={() => setSettingsDialogOpen(true)}
                  >
                    <SettingsIcon />
                    <span>{t("common.settings")}</span>
                  </SidebarMenuButton>
                </span>
              </TooltipTrigger>
              <TooltipContent side="top">{t("common.settings")}</TooltipContent>
            </Tooltip>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  tooltip={t("auth.logoutTooltip", {
                    username: user?.username ?? "",
                  })}
                  className="studio-sidebar-nav-button studio-sidebar-footer-button studio-sidebar-account-button"
                >
                  <CircleUserRoundIcon />
                  <span>{user?.username ?? t("auth.logout")}</span>
                  <ChevronUpIcon className="studio-sidebar-account-chevron" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                side="right"
                align="end"
                className="studio-sidebar-account-menu"
              >
                {isAdmin && (
                  <DropdownMenuItem
                    onSelect={() => setUserManagementDialogOpen(true)}
                  >
                    <UsersIcon />
                    <span>{t("auth.userManagement")}</span>
                  </DropdownMenuItem>
                )}
                <DropdownMenuItem
                  onSelect={() => setChangePasswordDialogOpen(true)}
                >
                  <KeyRoundIcon />
                  <span>{t("auth.changePassword")}</span>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  variant="destructive"
                  onSelect={() => {
                    void logout();
                  }}
                >
                  <LogOutIcon />
                  <span>{t("auth.logout")}</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>

      <SettingsDialog
        open={settingsDialogOpen}
        onOpenChange={setSettingsDialogOpen}
      />
      <WandbMonitorDialog
        open={wandbDialogOpen}
        onOpenChange={setWandbDialogOpen}
        wandbLinks={wandbLinks}
      />
      <UserManagementDialog
        open={userManagementDialogOpen}
        onOpenChange={setUserManagementDialogOpen}
      />
      <ChangePasswordDialog
        open={changePasswordDialogOpen}
        onOpenChange={setChangePasswordDialogOpen}
      />
    </Sidebar>
  );
};

export default memo(StudioSidebar);

