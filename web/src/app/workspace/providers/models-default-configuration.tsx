import { DefaultModelConfig, ModelConfig } from '@/api';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { apiClient } from '@/lib/api/client';
import _ from 'lodash';
import { Settings } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';

export const ModelsDefaultConfiguration = () => {
  const [defaultModels, setDefaultModels] = useState<DefaultModelConfig[]>([]);
  const [availableModels, setAvailableModels] = useState<ModelConfig[]>([]);
  const [visible, setVisible] = useState<boolean>(false);

  const router = useRouter();

  const getModels = useCallback(async () => {
    const [defaultModelsRes, availableModelsRes] = await Promise.all([
      apiClient.defaultApi.defaultModelsGet(),
      apiClient.defaultApi.availableModelsPost(),
    ]);
    setDefaultModels(defaultModelsRes.data.items || []);
    setAvailableModels(availableModelsRes.data.items || []);
  }, []);

  const handleSave = useCallback(async () => {}, []);

  useEffect(() => {
    if (visible) {
      getModels();
    }
  }, [getModels, visible]);

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="outline" onClick={() => setVisible(true)}>
            <Settings />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Default models configuration</TooltipContent>
      </Tooltip>
      <Dialog open={visible} onOpenChange={() => setVisible(false)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Default models configuration</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-6 py-8">
            {defaultModels.map((modelConfig) => {
              return (
                <div key={modelConfig.scenario} className="flex flex-col gap-2">
                  <Label>{_.startCase(modelConfig.scenario)}</Label>
                  <Select>
                    <SelectTrigger className="w-[280px]">
                      <SelectValue placeholder="Select a timezone" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectLabel>North America</SelectLabel>
                        <SelectItem value="est">
                          Eastern Standard Time (EST)
                        </SelectItem>
                        <SelectItem value="cst">
                          Central Standard Time (CST)
                        </SelectItem>
                        <SelectItem value="mst">
                          Mountain Standard Time (MST)
                        </SelectItem>
                        <SelectItem value="pst">
                          Pacific Standard Time (PST)
                        </SelectItem>
                        <SelectItem value="akst">
                          Alaska Standard Time (AKST)
                        </SelectItem>
                        <SelectItem value="hst">
                          Hawaii Standard Time (HST)
                        </SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>
              );
            })}
            <div></div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setVisible(false)}>
              Cancel
            </Button>
            <Button onClick={handleSave}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};
