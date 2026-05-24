"use client";

import { useEffect, useState } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

export function HealthCheckModal() {
  const [healthError, setHealthError] = useState(false);

  useEffect(() => {
    fetch("/api/health")
      .then((res) => {
        if (!res.ok) setHealthError(true);
      })
      .catch(() => setHealthError(true));
  }, []);

  return (
    <AlertDialog open={healthError}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>FastAPI接続エラー</AlertDialogTitle>
          <AlertDialogDescription>
            FastAPIに接続できません。AI回答生成機能が利用できない状態です。
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogAction onClick={() => window.location.reload()}>
            再読み込み
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
