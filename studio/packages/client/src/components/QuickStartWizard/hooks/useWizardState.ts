import { useState, useCallback } from 'react';
import { WizardState, DemoDataset } from '../types';

export function useWizardState() {
  // 从 sessionStorage 恢复初始步骤
  const savedStep = typeof window !== 'undefined' ? sessionStorage.getItem('wizard_step') : null;
  const initialStep = savedStep ? parseInt(savedStep, 10) : 0;

  const [state, setState] = useState<WizardState>({
    currentStep: initialStep,
    selectedDataset: null,
    isExecuting: false,
    executionStatus: 'idle',
    logs: [],
    canProceed: false,
  });

  const selectDataset = useCallback((dataset: DemoDataset) => {
    setState(prev => ({
      ...prev,
      selectedDataset: dataset,
      canProceed: true,
    }));
  }, []);

  const nextStep = useCallback(() => {
    setState(prev => {
      const newStep = Math.min(prev.currentStep + 1, 3);
      // 保存到 sessionStorage
      if (typeof window !== 'undefined') {
        sessionStorage.setItem('wizard_step', String(newStep));
      }
      return {
        ...prev,
        currentStep: newStep,
        canProceed: false,
        logs: [],
      };
    });
  }, []);

  const prevStep = useCallback(() => {
    setState(prev => ({
      ...prev,
      currentStep: Math.max(prev.currentStep - 1, 0),
      canProceed: true,
    }));
  }, []);

  const addLog = useCallback((message: string) => {
    setState(prev => ({
      ...prev,
      logs: [...prev.logs, `[${new Date().toLocaleTimeString()}] ${message}`],
    }));
  }, []);

  const setExecuting = useCallback((executing: boolean) => {
    setState(prev => ({
      ...prev,
      isExecuting: executing,
      executionStatus: executing ? 'running' : prev.executionStatus,
    }));
  }, []);

  const reset = useCallback(() => {
    setState({
      currentStep: 0,
      selectedDataset: null,
      isExecuting: false,
      executionStatus: 'idle',
      logs: [],
      canProceed: false,
    });
  }, []);

  return {
    state,
    selectDataset,
    nextStep,
    prevStep,
    addLog,
    setExecuting,
    reset,
  };
}
